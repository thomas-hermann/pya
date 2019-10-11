from .base import BackendBase, StreamBase

import asyncio
from IPython.display import Javascript, display
from sanic import Sanic


class JupyterBackend(BackendBase):

    dtype = 'float32'
    range = 1
    bs = 1024  # streaming introduces lack which has to be covered by the buffer

    def __init__(self, port=8765, proxy_suffix=None):
        self.dummy_devices = [dict(maxInputChannels=0, maxOutputChannels=2, index=0, name="JupyterBackend")]
        self.port = port
        self.proxy_suffix = proxy_suffix
        if self.proxy_suffix is not None:
            self.bs = 1024 * 10  # probably running on binder; increase buffer size

    def get_device_count(self):
        return len(self.dummy_devices)

    def get_device_info_by_index(self, idx):
        return self.dummy_devices[idx]

    def get_default_input_device_info(self):
        return self.dummy_devices[0]

    def get_default_output_device_info(self):
        return self.dummy_devices[0]

    def open(self, *args, channels, rate, stream_callback=None, **kwargs):
        return JupyterStream(channels=channels, rate=rate, stream_callback=stream_callback, port=self.port,
                             proxy_suffix=self.proxy_suffix)

    def process_buffer(self, buffer):
        return buffer

    def terminate(self):
        pass


class JupyterStream(StreamBase):

    def __init__(self, channels, rate, stream_callback, port, proxy_suffix):
        self.rate = rate
        self.channels = channels
        self.stream_callback = stream_callback
        self.cb_thread = None
        self.server = None
        self._is_active = False

        app = Sanic(__name__)

        async def bridge(request, ws):
            while True:
                req = await ws.recv()
                buffer = self.stream_callback(None, None, None, None)
                await ws.send(buffer.reshape(-1, 1, order='F').tobytes())

        app.add_websocket_route(bridge, '/')
        coro = app.create_server(host="0.0.0.0", port=8765, debug=False, access_log=False, return_asyncio_server=True)
        self.task = asyncio.ensure_future(coro)
        self.server = None

        url_suffix = f':{port}' if proxy_suffix is None else proxy_suffix

        self.client = Javascript(
            f"""
                var sampleRate = {self.rate};
                var channels = {self.channels};
                var urlSuffix = "{url_suffix}";
                window.pya = {{ bufferThresh: 0.2 }}
            """
            """
                var processedPackages = 0;
                var latePackages = 0;
                var badPackageRatio = 1;
                function resolve_proxy(request) {
                    var res = request;
                    var ps = window.location.pathname.split('/');
                    var idx = res.indexOf('*');
                    var i = 1;
                    while (idx > -1) {
                        res = res.replace('*', ps[i])
                        idx = res.indexOf('*')
                        i++;
                    }
                    return res
                }

                if (!window.audioContext) {
                    window.audioContext = new AudioContext();
                }

                var protocol = (window.location.protocol == 'https:') ? 'wss://' : 'ws://'
                var ws = new WebSocket(protocol+window.location.hostname+resolve_proxy(urlSuffix));
                ws.binaryType = 'arraybuffer';
                window.ws = ws;
                var startTime = 0;
                window.AudioContext =  window.AudioContext = window.AudioContext||window.webkitAudioContext;
                var context = new AudioContext();

                ws.onopen = function() {
                    console.log("PyaJSClient: Websocket connected.");
                    startTime = context.currentTime;
                    ws.send("G");
                };

                ws.onmessage = function (evt) {
                    if (evt.data) {
                        processedPackages++;
                        var buf = new Float32Array(evt.data)
                        var duration = buf.length / channels
                        var buffer = context.createBuffer(channels, duration, sampleRate)
                        for (let i = 0; i < channels; i++) {
                            buffer.copyToChannel(buf.slice(i*duration, (i+1) * duration), i, 0)
                        }
                        var source = context.createBufferSource()
                        source.buffer = buffer
                        source.connect(context.destination)
                        if (startTime > context.currentTime) {
                            source.start(startTime)
                            startTime += buffer.duration
                        } else {
                            latePackages++;
                            badPackageRatio = latePackages / processedPackages
                            if (processedPackages > 50) {
                                console.log("PyaJSClient: Dropped sample ratio is " + badPackageRatio.toFixed(2))
                                if (badPackageRatio > 0.05) {
                                    let tr = window.pya.bufferThresh
                                    window.pya.bufferThresh = (tr > 0.01) ? tr - 0.03 : 0.01;
                                    console.log("PyaJSClient: Decrease buffer delay to " + window.pya.bufferThresh.toFixed(2))
                                }
                                latePackages = 0;
                                processedPackages = 0;
                            }
                            source.start()
                            startTime = context.currentTime + buffer.duration
                        }
                        setTimeout(function() {ws.send("G")},
                                   (startTime - context.currentTime) * 1000 * window.pya.bufferThresh)
                    }
                };
                console.log("PyaJSClient: Websocket client loaded.")
            """)

    @staticmethod
    def set_buffer_threshold(buffer_limit):
        display(Javascript(f"window.pya.bufferThresh = {1 - buffer_limit}"))

    def stop_stream(self):
        self._is_active = False

    async def start_server(self):
        self.server = await self.task
        display(self.client)
        self._is_active = True

    def close(self):
        self.server.close()

    def start_stream(self):
        asyncio.create_task(self.start_server())

    def is_active(self):
        return self._is_active


