#!/usr/bin/env python3
"""Prompt page for LFM3-Omni-2B, served from the IQ-9075 on 127.0.0.1:8080.

GET  /                -> the page
POST /api/generate    -> body: one JSON line {"prompt", "mode", "image_len", "audio_len"},
                         then image_len bytes of JPEG and audio_len bytes of 16 kHz mu-law.
                         Returns {"id", "segments", "text", "tokens", "seconds", "note"};
                         audio segments are [offset, length] into the answer's audio.
GET  /api/audio/<id>  -> the answer's audio, 16 kHz mu-law

Media is compact on purpose: it crosses a 921600-baud serial link. Standard library only.
"""
import itertools, json, threading, time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import omni

AUDIO = OrderedDict()  # answer id -> mu-law bytes, last few answers only
IDS = itertools.count(1)
MODEL_LOCK = threading.Lock()

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LFM3-Omni on IQ-9075</title>
<style>
  :root { color-scheme: light dark; --bg:#fafafa; --fg:#1a1a1a; --muted:#666; --line:#ddd; --chip:#eee; --accent:#2563eb; --rec:#dc2626; }
  @media (prefers-color-scheme: dark) { :root { --bg:#141414; --fg:#eee; --muted:#999; --line:#333; --chip:#262626; --accent:#60a5fa; --rec:#f87171; } }
  body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; }
  main { max-width:760px; margin:0 auto; padding:32px 16px; }
  h1 { font-size:20px; margin:0 0 4px; }
  p.sub { margin:0 0 20px; color:var(--muted); font-size:14px; }
  textarea { width:100%; box-sizing:border-box; min-height:110px; padding:12px; font:inherit;
             background:transparent; color:inherit; border:1px solid var(--line); border-radius:8px; resize:vertical; }
  .row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:10px; }
  button, label.btn { padding:8px 14px; font:inherit; font-size:14px; border:1px solid var(--line); border-radius:8px;
           background:var(--chip); color:inherit; cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; padding:9px 18px; }
  button.recording { background:var(--rec); border-color:var(--rec); color:#fff; }
  button:disabled { opacity:.5; cursor:wait; }
  .chip { display:inline-flex; gap:8px; align-items:center; padding:4px 6px 4px 10px; border-radius:999px; background:var(--chip); font-size:14px; }
  .chip img { height:28px; border-radius:4px; }
  .chip button { padding:0 8px; border:0; background:transparent; }
  fieldset { border:0; padding:0; margin:14px 0 0; display:flex; gap:16px; flex-wrap:wrap; font-size:14px; }
  legend { color:var(--muted); font-size:13px; margin-bottom:4px; }
  #out { margin-top:24px; white-space:pre-wrap; padding:16px; border:1px solid var(--line); border-radius:8px; min-height:2em; }
  #out .speaking { border-bottom:2px solid var(--accent); }
  #meta { margin-top:8px; color:var(--muted); font-size:13px; }
  .hidden { display:none !important; }
</style></head>
<body><main>
  <h1>LFM3-Omni-2B</h1>
  <p class="sub">Running on the Dragonwing IQ-9075 (llama.cpp, CPU). Ctrl/Cmd+Enter to send.</p>
  <textarea id="prompt" placeholder="Type a prompt, record your voice, attach an image, or combine them..." autofocus></textarea>
  <div class="row">
    <button id="rec" type="button">Record voice</button>
    <label class="btn" for="file">Attach image</label>
    <input id="file" type="file" accept="image/*" class="hidden">
    <span id="voiceChip" class="chip hidden"><span id="voiceLabel"></span><button type="button" id="voicePlay" title="Play">&#9654;</button><button type="button" id="voiceDel" title="Remove">&times;</button></span>
    <span id="imgChip" class="chip hidden"><img id="imgThumb" alt=""><span id="imgLabel"></span><button type="button" id="imgDel" title="Remove">&times;</button></span>
  </div>
  <fieldset>
    <legend>Response</legend>
    <label><input type="radio" name="mode" value="text" checked> Text</label>
    <label><input type="radio" name="mode" value="audio"> Voice</label>
    <label><input type="radio" name="mode" value="interleaved"> Text + voice (interleaved)</label>
  </fieldset>
  <div class="row"><button id="go" class="primary">Send</button><button id="replay" class="hidden">Replay</button></div>
  <div id="out"></div>
  <div id="meta"></div>
<script>
const $ = id => document.getElementById(id);
const RATE = 16000;
let voice = null, image = null, recorder = null, recStart = 0, last = null, ctx = null, playing = null;

// ---- mu-law (G.711), matching Python's audioop ----
function ulawEncode(f32) {
  const out = new Uint8Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    let s = Math.round(Math.max(-1, Math.min(1, f32[i])) * 32767);
    const sign = s < 0 ? 0x80 : 0; if (sign) s = -s;
    s = Math.min(s, 32635) + 0x84;
    let exp = 7; for (let m = 0x4000; (s & m) === 0 && exp > 0; exp--, m >>= 1);
    out[i] = ~(sign | (exp << 4) | ((s >> (exp + 3)) & 0x0F)) & 0xFF;
  }
  return out;
}
function ulawDecode(u8) {
  const out = new Float32Array(u8.length);
  for (let i = 0; i < u8.length; i++) {
    const u = ~u8[i] & 0xFF, exp = (u >> 4) & 7;
    const s = ((((u & 0x0F) << 3) + 0x84) << exp) - 0x84;
    out[i] = (u & 0x80 ? -s : s) / 32768;
  }
  return out;
}
function audioCtx() { ctx = ctx || new AudioContext(); if (ctx.state === "suspended") ctx.resume(); return ctx; }

// ---- voice recording: record, then resample to 16 kHz mono mu-law ----
$("rec").onclick = async () => {
  if (recorder) { recorder.stop(); return; }
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
  catch (e) { $("meta").textContent = "Microphone unavailable: " + e.message; return; }
  const chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = e => chunks.push(e.data);
  recorder.onstop = async () => {
    stream.getTracks().forEach(t => t.stop());
    recorder = null; $("rec").classList.remove("recording"); $("rec").textContent = "Record voice";
    const buf = await audioCtx().decodeAudioData(await new Blob(chunks).arrayBuffer());
    const off = new OfflineAudioContext(1, Math.ceil(buf.duration * RATE), RATE);
    const src = off.createBufferSource(); src.buffer = buf; src.connect(off.destination); src.start();
    const pcm = (await off.startRendering()).getChannelData(0);
    voice = { ulaw: ulawEncode(pcm), seconds: buf.duration };
    $("voiceLabel").textContent = "Voice " + voice.seconds.toFixed(1) + " s";
    $("voiceChip").classList.remove("hidden");
  };
  recorder.start(); recStart = Date.now();
  $("rec").classList.add("recording");
  const tick = () => { if (!recorder) return; $("rec").textContent = "Stop (" + ((Date.now() - recStart) / 1000).toFixed(0) + " s)"; setTimeout(tick, 250); };
  tick();
};
$("voicePlay").onclick = () => voice && play([{ type: "audio", offset: 0, length: voice.ulaw.length }], voice.ulaw, false);
$("voiceDel").onclick = () => { voice = null; $("voiceChip").classList.add("hidden"); };

// ---- image: downscale to 768 px JPEG ----
$("file").onchange = async () => {
  const f = $("file").files[0]; $("file").value = "";
  if (!f) return;
  const bmp = await createImageBitmap(f);
  const k = Math.min(1, 768 / Math.max(bmp.width, bmp.height));
  const c = document.createElement("canvas"); c.width = Math.round(bmp.width * k); c.height = Math.round(bmp.height * k);
  c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
  const blob = await new Promise(r => c.toBlob(r, "image/jpeg", 0.85));
  image = new Uint8Array(await blob.arrayBuffer());
  $("imgThumb").src = URL.createObjectURL(blob);
  $("imgLabel").textContent = Math.round(image.length / 1024) + " KB";
  $("imgChip").classList.remove("hidden");
};
$("imgDel").onclick = () => { image = null; $("imgChip").classList.add("hidden"); };

// ---- playback: text appears and audio plays in the order the model produced them ----
async function play(segments, ulaw, showText = true) {
  const token = playing = {};
  const ac = audioCtx(), out = $("out");
  if (showText) out.textContent = "";
  for (const seg of segments) {
    if (playing !== token) return;
    if (seg.type === "text") { if (showText) out.append(seg.text); continue; }
    const f32 = ulawDecode(ulaw.subarray(seg.offset, seg.offset + seg.length));
    const b = ac.createBuffer(1, f32.length, RATE); b.copyToChannel(f32, 0);
    const src = ac.createBufferSource(); src.buffer = b; src.connect(ac.destination);
    const spoken = out.lastChild; if (spoken && spoken.nodeType === 3 && showText) {
      const span = document.createElement("span"); span.className = "speaking"; span.textContent = spoken.textContent; spoken.replaceWith(span);
    }
    await new Promise(r => { src.onended = r; src.start(); });
    const s = out.querySelector(".speaking"); if (s) s.replaceWith(document.createTextNode(s.textContent));
  }
}
$("replay").onclick = () => last && play(last.segments, last.ulaw, last.mode !== "audio");

// ---- send ----
async function send() {
  const prompt = $("prompt").value.trim();
  if (!prompt && !voice && !image) return;
  const mode = document.querySelector("input[name=mode]:checked").value;
  audioCtx(); playing = null;
  $("go").disabled = true; $("replay").classList.add("hidden"); $("meta").textContent = "";
  $("out").textContent = (voice || image) ? "Sending media to the device and generating..." : "Generating on the device...";
  const t0 = Date.now();
  try {
    const head = new TextEncoder().encode(JSON.stringify({ prompt, mode,
      image_len: image ? image.length : 0, audio_len: voice ? voice.ulaw.length : 0 }) + "\n");
    const body = new Blob([head, image || new Uint8Array(), voice ? voice.ulaw : new Uint8Array()]);
    const r = await fetch("/api/generate", { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || r.statusText);
    let ulaw = new Uint8Array();
    if (d.audio_bytes) {
      $("out").textContent = "Downloading " + Math.round(d.audio_bytes / 1024) + " KB of speech...";
      ulaw = new Uint8Array(await (await fetch("/api/audio/" + d.id)).arrayBuffer());
    }
    last = { segments: d.segments, ulaw, mode };
    const secs = d.segments.filter(s => s.type === "audio").reduce((a, s) => a + s.length, 0) / RATE;
    $("meta").textContent = d.tokens + " tokens in " + d.seconds + " s on the device" +
      (secs ? ", " + secs.toFixed(1) + " s of speech" : "") + " | " + ((Date.now() - t0) / 1000).toFixed(1) + " s total" +
      (d.note ? " | " + d.note : "");
    if (mode === "audio") $("out").textContent = d.text ? "" : "(speaking)";
    if (d.audio_bytes) $("replay").classList.remove("hidden");
    $("go").disabled = false;
    if (!d.segments.length) $("out").textContent = "(empty response)";
    else await play(d.segments, ulaw, mode !== "audio");
  } catch (e) {
    $("out").textContent = "Error: " + e.message;
  } finally { $("go").disabled = false; }
}
$("go").onclick = send;
$("prompt").addEventListener("keydown", e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) send(); });
</script>
</main></body></html>
"""


def generate(meta, image, ulaw):
    mode = meta.get("mode") if meta.get("mode") in omni.MODALITIES else "text"
    prompt = (meta.get("prompt") or "").strip()
    pcm = omni.from_wire(ulaw) if ulaw else None
    start = time.time()
    with MODEL_LOCK:
        if image:
            ans = omni.with_image(prompt, mode, image, pcm)
        else:
            ans = omni.realtime(prompt, mode, pcm)

    segments, audio = [], b""
    for kind, data in ans.segments:
        if kind == "text":
            segments.append({"type": "text", "text": data})
        else:
            wire = omni.to_wire(data, ans.rate)
            segments.append({"type": "audio", "offset": len(audio), "length": len(wire)})
            audio += wire
    answer_id = str(next(IDS))
    if audio:
        AUDIO[answer_id] = audio
        while len(AUDIO) > 5:
            AUDIO.popitem(last=False)
    return {"id": answer_id, "segments": segments, "text": ans.text, "audio_bytes": len(audio),
            "tokens": ans.tokens, "seconds": round(time.time() - start, 1), "note": ans.note}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, ctype, data):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status, obj):
        self._send(status, "application/json", json.dumps(obj).encode())

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif self.path.startswith("/api/audio/") and self.path[11:] in AUDIO:
            self._send(200, "application/octet-stream", AUDIO[self.path[11:]])
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/generate":
            return self._json(404, {"error": "not found"})
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            head, _, media = body.partition(b"\n")
            meta = json.loads(head or b"{}")
            n_img = int(meta.get("image_len") or 0)
            image, ulaw = media[:n_img], media[n_img:n_img + int(meta.get("audio_len") or 0)]
            if not (meta.get("prompt") or "").strip() and not image and not ulaw:
                return self._json(400, {"error": "empty prompt"})
            self._json(200, generate(meta, image, ulaw))
        except Exception as e:
            self._json(502, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8080), Handler).serve_forever()
