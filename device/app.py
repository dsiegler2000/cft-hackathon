#!/usr/bin/env python3
"""Prompt page for LFM3-Omni-2B, served from the IQ-9075 on 127.0.0.1:8080.

GET  /                -> the page
POST /api/generate    -> body: one JSON line {"prompt", "mode", "image_len", "audio_len"},
                         then image_len bytes of JPEG and audio_len bytes of 16 kHz mu-law.
                         Returns {"id", "segments", "text", "level", "concern", "tokens", "seconds", "note"};
                         audio segments are [offset, length] into the answer's audio.
                         The input goes in with the triage instructions; "level" is RED, YELLOW,
                         GREEN or null, and "concern" is the model's Concern line.
GET  /api/audio/<id>  -> the answer's audio, 16 kHz mu-law

Media is compact on purpose: it crosses a 921600-baud serial link. Standard library only.
"""
import itertools, json, re, threading, time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import omni

AUDIO = OrderedDict()  # answer id -> mu-law bytes, last few answers only
IDS = itertools.count(1)
MODEL_LOCK = threading.Lock()

# Triage instructions, sent as text in the same user turn as the voice (after it), which is
# how they were tested on the board. Without them the model restates what it was told.
# Tested with LFM3-Omni-2B: the placeholders keep the format without being copied, an
# example answer gets copied, and the "do not repeat" line stops the restating. The model
# still misjudges some levels (chest pain as YELLOW), so the page raises the level to RED
# when its fixed red-flag rules match the notes or the Concern line.
TRIAGE = """You are a medical triage assistant for first responders. The responder describes a patient. Do not repeat, restate or summarize the description; the responder already knows it. Do not ask questions. Never claim a diagnosis.
Reply in exactly this format, four short lines and nothing else:
Concern: <the medical problem you suspect, in a few words>
Check: <two or three things to check on this patient>
Do now: <two or three first actions for this patient>
TRIAGE: <RED or YELLOW or GREEN>
RED means life-threatening now, YELLOW means needs care within hours, GREEN means minor. Anything that threatens breathing, the heart, the brain or consciousness is RED. When unsure, choose the higher level. Only advise CPR if the patient is not breathing."""
# A TRIAGE line naming exactly one level (an echoed "RED or YELLOW or GREEN" does not count),
# or a line that is only the level, which the model sometimes writes instead.
LEVEL = re.compile(r"^[^\w\n]*(?:TRIAGE(?: level)?[^\w\n]*(RED|YELLOW|GREEN)\b"
                   r"(?![^\n]*\b(?-i:RED|YELLOW|GREEN)\b)[^\n]*|(RED|YELLOW|GREEN)[^\w\n]*)$", re.I | re.M)
CONCERN = re.compile(r"^[^\w\n]*Concern[^\w\n]*(.+)$", re.I | re.M)


def triage_prompt(prompt):
    """The triage instructions, then the responder's typed notes, if any."""
    return TRIAGE + ("\n\nCase: " + prompt if prompt else "")

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
  textarea { width:100%; box-sizing:border-box; min-height:70px; margin-top:14px; padding:12px; font:inherit;
             background:transparent; color:inherit; border:1px solid var(--line); border-radius:8px; resize:vertical; }
  .row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:10px; }
  button, label.btn { padding:8px 14px; font:inherit; font-size:14px; border:1px solid var(--line); border-radius:8px;
           background:var(--chip); color:inherit; cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; padding:9px 18px; }
  button.talk { font-size:17px; padding:12px 32px; }
  button.recording { background:var(--rec); border-color:var(--rec); color:#fff; }
  button:disabled { opacity:.5; cursor:wait; }
  .chip { display:inline-flex; gap:8px; align-items:center; padding:4px 6px 4px 10px; border-radius:999px; background:var(--chip); font-size:14px; }
  .chip img { height:28px; border-radius:4px; }
  .chip button { padding:0 8px; border:0; background:transparent; }
  .hint { color:var(--muted); font-size:13px; }
  .alarm { margin-top:16px; padding:10px 14px; border-radius:8px; background:var(--rec); color:#fff; font-weight:600; }
  .level { margin-top:24px; padding:10px 16px; border-radius:8px; font-size:18px; font-weight:700; color:#fff; }
  .level.red { background:#dc2626; } .level.yellow { background:#eab308; color:#1a1a1a; } .level.green { background:#16a34a; }
  .level.none { background:var(--chip); color:var(--muted); font-size:14px; font-weight:400; }
  .level:not(.hidden) + #out { margin-top:8px; }
  #out { margin-top:24px; white-space:pre-wrap; padding:16px; border:1px solid var(--line); border-radius:8px; min-height:2em; }
  #out .speaking { border-bottom:2px solid var(--accent); }
  #meta { margin-top:8px; color:var(--muted); font-size:13px; }
  .hidden { display:none !important; }
</style></head>
<body><main>
  <h1>Triage assistant</h1>
  <p class="sub">Describe the patient by voice; the answer comes back spoken and written. LFM3-Omni-2B, offline on the Dragonwing IQ-9075. Decision support only, not a diagnosis.</p>
  <div class="row"><button id="talk" type="button" class="primary talk">Talk</button><span id="cue" class="hint">Tap, describe the patient, tap again to send.</span></div>
  <div class="row" id="scenarios"><span class="hint">Start with:</span>
    <button type="button" data-topic="Wound" data-photo data-cue="Add a photo, then tap Talk: where it is, how big, how it happened.">Wound</button>
    <button type="button" data-topic="Burn" data-photo data-cue="Add a photo, then tap Talk: where, how big, what caused it.">Burn</button>
    <button type="button" data-topic="Skin rash" data-photo data-cue="Add a photo, then tap Talk: where, how long, any fever or itching.">Rash</button>
    <button type="button" data-topic="Breathing problem" data-cue="Say how fast they breathe, any wheeze, their lip colour. Tap again to send.">Breathing</button>
    <button type="button" data-topic="Head injury" data-cue="Say how it happened, whether they were knocked out, vomiting or confused. Tap again to send.">Head injury</button>
  </div>
  <textarea id="prompt" placeholder="Optional notes, typed instead of or as well as speaking. Ctrl/Cmd+Enter to send."></textarea>
  <div class="row">
    <label class="btn" for="file">Add photo</label>
    <input id="file" type="file" accept="image/*" class="hidden">
    <span id="voiceChip" class="chip hidden"><span id="voiceLabel"></span><button type="button" id="voicePlay" title="Play">&#9654;</button><button type="button" id="voiceDel" title="Remove">&times;</button></span>
    <span id="imgChip" class="chip hidden"><img id="imgThumb" alt=""><span id="imgLabel"></span><button type="button" id="imgDel" title="Remove">&times;</button></span>
    <button id="go">Send</button><button id="replay" class="hidden">Replay</button>
  </div>
  <div id="alarm" class="alarm hidden" role="alert"></div>
  <div id="level" class="level hidden"></div>
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

// ---- voice: tap Talk, speak, tap again; the recording (16 kHz mono mu-law) is sent with any notes and photo ----
async function talk() {
  if (recorder) { recorder.stop(); return; }
  if ($("go").disabled) return;
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
  catch (e) { $("meta").textContent = "Microphone unavailable: " + e.message; return; }
  const chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = e => chunks.push(e.data);
  recorder.onstop = async () => {
    stream.getTracks().forEach(t => t.stop());
    recorder = null; $("talk").classList.remove("recording"); $("talk").textContent = "Talk";
    if (Date.now() - recStart < 600) return;  // an accidental tap, not a description
    const buf = await audioCtx().decodeAudioData(await new Blob(chunks).arrayBuffer());
    const off = new OfflineAudioContext(1, Math.ceil(buf.duration * RATE), RATE);
    const src = off.createBufferSource(); src.buffer = buf; src.connect(off.destination); src.start();
    const pcm = (await off.startRendering()).getChannelData(0);
    voice = { ulaw: ulawEncode(pcm), seconds: buf.duration };
    $("voiceLabel").textContent = "Voice " + voice.seconds.toFixed(1) + " s";
    $("voiceChip").classList.remove("hidden");
    send();
  };
  recorder.start(); recStart = Date.now();
  $("talk").classList.add("recording");
  const tick = () => { if (!recorder) return; $("talk").textContent = "Stop and send (" + ((Date.now() - recStart) / 1000).toFixed(0) + " s)"; setTimeout(tick, 250); };
  tick();
}
$("talk").onclick = talk;

// ---- scenario buttons: name the case and say what to describe ----
document.querySelectorAll("#scenarios button").forEach(b => b.onclick = () => {
  $("prompt").value = b.dataset.topic + ". "; $("cue").textContent = b.dataset.cue; checkFlags();
  if ("photo" in b.dataset) $("file").click(); else if (!recorder) talk();
});

// ---- red flags: fixed rules, independent of the model's judgement, on the typed notes and the model's Concern line ----
const RED_FLAGS = [
  ["chest pain", /chest (pain|pressure|tightness)/i],
  ["possible heart attack", /heart attack|cardiac|myocardial|coronary|angina/i],
  ["trouble breathing", /(can'?t|cannot|unable to|struggling to|not) breath|short(ness)? of breath|difficulty breathing|breathing difficult|respiratory (distress|failure)|airway (obstruct|block)|choking|gasping/i],
  ["unresponsive", /unconscious|unresponsive|passed out|not waking|won'?t wake/i],
  ["severe bleeding", /(severe|heavy|uncontrolled|massive|spurting) bleed|bleeding (heavily|a lot|badly|won'?t stop)|haemorrhag|hemorrhag/i],
  ["sepsis", /sepsis|septic/i],
  ["stroke signs", /stroke|face (is )?droop|drooping face|slurred speech|one side (is )?(weak|numb)/i],
  ["seizure", /seizure|convuls/i],
  ["allergic reaction", /anaphyla|throat (is )?(swelling|closing|tight)|swollen (tongue|throat)/i],
  ["blue lips or skin", /blue (lips|skin|face)|cyano/i],
  ["no pulse", /no pulse|cardiac arrest|heart (has )?stopped/i],
  ["overdose", /overdos/i],
  ["self-harm risk", /suicid|kill (myself|himself|herself|themsel)/i],
  ["sudden severe headache", /worst headache|thunderclap/i],
];
function beep() {
  const ac = audioCtx(), o = ac.createOscillator(), g = ac.createGain();
  o.frequency.value = 880; g.gain.value = 0.2; o.connect(g).connect(ac.destination); o.start(); o.stop(ac.currentTime + 0.4);
}
function checkFlags(concern = "") {
  const text = $("prompt").value + "\n" + concern;
  const hits = RED_FLAGS.filter(([, re]) => re.test(text)).map(([name]) => name);
  const wasHidden = $("alarm").classList.contains("hidden");
  $("alarm").textContent = "Possible emergency: " + hits.join(", ") + ". Get emergency help now; do not wait for the assistant.";
  $("alarm").classList.toggle("hidden", !hits.length);
  if (hits.length && wasHidden) beep();
  return hits;
}
$("prompt").addEventListener("input", () => checkFlags());

// ---- triage level badge: a red flag raises the model's level to RED ----
const LEVELS = { RED: "RED: life-threatening, act now", YELLOW: "YELLOW: urgent, care within hours", GREEN: "GREEN: minor" };
function showLevel(level, flagged) {
  const raised = flagged && level !== "RED";
  if (flagged) level = "RED";
  $("level").textContent = level ? "Triage " + LEVELS[level] + (raised ? " (raised by red-flag rules)" : "") : "No triage level in the answer";
  $("level").className = "level " + (level ? level.toLowerCase() : "none");
}
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
  if (recorder) return recorder.stop();  // Send while talking: finish the recording, which sends
  const prompt = $("prompt").value.trim();
  if (!prompt && !voice && !image) return;
  if ($("go").disabled) return;
  const mode = "interleaved";  // answers are always spoken and written
  audioCtx(); playing = null; checkFlags();
  $("go").disabled = $("talk").disabled = true; $("replay").classList.add("hidden"); $("level").classList.add("hidden"); $("meta").textContent = "";
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
    const flagged = checkFlags(d.concern || "").length > 0;
    if (d.text) showLevel(d.level, flagged);
    if (mode === "audio") $("out").textContent = d.text ? "" : "(speaking)";
    if (d.audio_bytes) $("replay").classList.remove("hidden");
    $("go").disabled = $("talk").disabled = false;
    if (!d.segments.length) $("out").textContent = "(empty response)";
    else await play(d.segments, ulaw, mode !== "audio");
  } catch (e) {
    $("out").textContent = "Error: " + e.message;
  } finally { $("go").disabled = $("talk").disabled = false; }
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
            ans = omni.with_image(triage_prompt(prompt), mode, image, pcm)
        else:
            ans = omni.realtime(triage_prompt(prompt), mode, pcm)
    levels, concern = LEVEL.findall(ans.text), CONCERN.search(ans.text)

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
            "level": "".join(levels[-1]).upper() if levels else None,
            "concern": concern and concern.group(1).strip(),
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
