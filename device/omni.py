"""LFM3-Omni-2B client for the local llama-server (127.0.0.1:8081). Standard library only.

Text and voice input go through the /v1/realtime WebSocket, where the model itself answers
in text, speech, or interleaved text and speech. Image input goes through
/v1/chat/completions (text out); for speech modes that answer is then spoken with /tts.

Audio inside this module is PCM16 mono bytes; to_wire()/from_wire() convert to the
16 kHz mu-law used on the slow serial link.
"""
import base64, json, os, re, socket, struct, urllib.request, warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop  # removed in Python 3.13; the board runs 3.12

HOST, PORT = "127.0.0.1", 8081
WIRE_RATE = 16000
MODALITIES = {"text": ["text"], "audio": ["audio"], "interleaved": ["text", "audio"]}
MAX_OUTPUT_TOKENS = 1024  # text tokens plus 80 ms audio frames


# ---- audio helpers -------------------------------------------------------------------

def to_wire(pcm, rate):
    """PCM16 mono at `rate` -> 16 kHz mu-law."""
    if rate != WIRE_RATE:
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, WIRE_RATE, None)
    return audioop.lin2ulaw(pcm, 2)


def from_wire(ulaw):
    """16 kHz mu-law -> PCM16 mono at 16 kHz."""
    return audioop.ulaw2lin(ulaw, 2)


def wav(pcm, rate):
    hdr = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", len(pcm))
    return hdr + pcm


# ---- minimal WebSocket client --------------------------------------------------------

class WebSocket:
    def __init__(self, path, timeout=300):
        self.sock = socket.create_connection((HOST, PORT), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {HOST}:{PORT}\r\nUpgrade: websocket\r\n"
                           f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                           "Sec-WebSocket-Version: 13\r\n\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("websocket handshake closed")
            head += chunk
        head, self.buf = head.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise ConnectionError("websocket handshake failed: " + head.split(b"\r\n", 1)[0].decode())

    def _read(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("websocket closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _send_frame(self, opcode, data):
        mask = os.urandom(4)
        n = len(data)
        hdr = bytes([0x80 | opcode])
        if n < 126:
            hdr += bytes([0x80 | n])
        elif n < 65536:
            hdr += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            hdr += bytes([0x80 | 127]) + struct.pack(">Q", n)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        self.sock.sendall(hdr + mask + masked)

    def send(self, obj):
        self._send_frame(0x1, json.dumps(obj).encode())

    def recv(self):
        """Next JSON message; answers pings, joins fragments."""
        msg = b""
        while True:
            b0, b1 = self._read(2)
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            data = self._read(n)
            op = b0 & 0x0F
            if op == 0x9:
                self._send_frame(0xA, data)
            elif op == 0x8:
                raise ConnectionError("websocket closed by server")
            elif op in (0x0, 0x1, 0x2):
                msg += data
                if b0 & 0x80:
                    return json.loads(msg)

    def close(self):
        try:
            self._send_frame(0x8, b"")
        finally:
            self.sock.close()


# ---- model calls ---------------------------------------------------------------------

class Answer:
    """Ordered text/audio segments of one response, audio kept as PCM16 at `rate`."""

    def __init__(self, rate):
        self.rate, self.segments, self.tokens, self.note = rate, [], 0, None

    def add(self, kind, data):
        if self.segments and self.segments[-1][0] == kind:
            self.segments[-1][1] += data
        else:
            self.segments.append([kind, data])

    @property
    def text(self):
        return "".join(d for k, d in self.segments if k == "text").strip()


def realtime(prompt, mode, pcm=None, pcm_rate=WIRE_RATE, system=None):
    """Text and/or voice in, answer in `mode` (text, audio or interleaved). `system` becomes
    the session instructions, which the server places after the model's own speech prompt."""
    ws = WebSocket("/v1/realtime")
    try:
        created = ws.recv()
        if created.get("type") == "error":
            raise RuntimeError(created["error"]["message"])
        out_rate = created["session"]["audio"]["output"]["format"]["rate"]
        session = {"type": "realtime", "output_modalities": MODALITIES[mode],
                   "max_output_tokens": MAX_OUTPUT_TOKENS,
                   "audio": {"input": {"format": {"type": "audio/pcm", "rate": pcm_rate}}}}
        if system:
            session["instructions"] = system
        ws.send({"type": "session.update", "session": session})
        if pcm:
            step = pcm_rate * 2  # one second per chunk
            for i in range(0, len(pcm), step):
                ws.send({"type": "input_audio_buffer.append",
                         "audio": base64.b64encode(pcm[i:i + step]).decode()})
            ws.send({"type": "input_audio_buffer.commit"})
        if prompt:
            ws.send({"type": "conversation.item.create", "item": {
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": prompt}]}})
        ws.send({"type": "response.create"})

        ans = Answer(out_rate)
        while True:
            ev = ws.recv()
            t = ev.get("type")
            if t == "response.output_text.delta":
                ans.add("text", ev["delta"])
            elif t == "response.output_audio.delta":
                ans.add("audio", base64.b64decode(ev["delta"]))
            elif t == "error":
                raise RuntimeError(ev["error"]["message"])
            elif t == "response.done":
                ans.tokens = (ev["response"].get("usage") or {}).get("output_tokens", 0)
                return ans
    finally:
        ws.close()


def _post(path, body, timeout=300):
    req = urllib.request.Request(f"http://{HOST}:{PORT}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", "")


def speak(text):
    """Text -> (PCM16, rate) with the model's own voice."""
    data, ctype = _post("/tts", {"input": text, "response_format": "pcm16", "n_predict": 1024})
    rate = int(next(p.split("=")[1] for p in ctype.split(";") if p.strip().startswith("rate=")))
    return data, rate


def with_image(prompt, mode, jpeg, pcm=None, pcm_rate=WIRE_RATE, system=None):
    """Image (plus optional text and voice) in. The model answers in text; speech modes
    then read that answer aloud with /tts, since chat completions cannot return audio."""
    content = [{"type": "image_url", "image_url": {
        "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}]
    if pcm:
        content.append({"type": "input_audio", "input_audio": {
            "data": base64.b64encode(wav(pcm, pcm_rate)).decode(), "format": "wav"}})
    content.append({"type": "text", "text": prompt or (
        "Answer the question in the audio about this image." if pcm else "Describe this image.")})
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": content})
    data, _ = _post("/v1/chat/completions", {"messages": messages, "max_tokens": 512, "stream": False})
    d = json.loads(data)
    text = (d["choices"][0]["message"].get("content") or "").strip()

    ans = Answer(WIRE_RATE)
    ans.tokens = d.get("usage", {}).get("completion_tokens", 0)
    ans.add("text", text)
    if mode != "text" and text:
        pcm_out, ans.rate = speak(re.sub(r"[*_#`>|]+", "", text))
        if mode == "audio":
            ans.segments = []
        ans.add("audio", pcm_out)
        ans.note = "With an image, the model answers in text and the speech is generated from that text."
    return ans
