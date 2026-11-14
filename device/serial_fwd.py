#!/usr/bin/env python3
"""Board side of the serial relay: replay one HTTP request against the local app.

Usage (typed on the UART console by laptop/serial_relay.py):
    python3 serial_fwd.py TAG META_B64 BODY_LEN BODY_SHA256

META_B64 is base64 JSON {"method", "path", "headers"}. The script switches the console to
raw mode, prints READY_<TAG>, then reads BODY_LEN bytes of base64 body (the relay sends
padding after it, because the UART holds back the tail of a burst until more bytes arrive).
It prints __R<TAG>__, then "H <base64 JSON status/headers>", the body as base64 lines,
and "Z <sha256 of the body>".
"""
import base64, hashlib, http.client, json, os, select, sys, termios, time, tty


def read_body(n, idle=3):
    """Read exactly n bytes from the console in raw mode, then drop the padding.
    Gives up after `idle` seconds without input, so a relay that died mid-upload
    does not leave the console stuck in raw mode."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setraw(fd)
    data = b""
    try:
        print(f"READY_{TAG}", end="\r\n", flush=True)
        while len(data) < n and select.select([fd], [], [], idle)[0]:
            data += os.read(fd, n - len(data))
    finally:
        time.sleep(0.2)
        termios.tcsetattr(fd, termios.TCSAFLUSH, old)
    return data


def reply(status, headers, body):
    print(f"__R{TAG}__")
    print("H " + base64.b64encode(json.dumps({"status": status, "headers": headers}).encode()).decode())
    sys.stdout.write(base64.encodebytes(body).decode())
    print("Z " + hashlib.sha256(body).hexdigest(), flush=True)


TAG, meta_b64, n, want = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
meta = json.loads(base64.b64decode(meta_b64))
body_b64 = read_body(n)
body = base64.b64decode(body_b64) if body_b64 else b""
if len(body_b64) != n or hashlib.sha256(body).hexdigest() != want:
    reply(599, [["Content-Type", "text/plain"]], f"upload corrupted ({len(body_b64)}/{n} bytes)".encode())
    sys.exit(0)

conn = http.client.HTTPConnection("127.0.0.1", 8080, timeout=600)
try:
    conn.request(meta["method"], meta["path"], body, meta["headers"])
    r = conn.getresponse()
    reply(r.status, r.getheaders(), r.read())
except Exception as e:
    reply(502, [["Content-Type", "application/json"]],
          json.dumps({"error": f"device app unreachable: {e}"}).encode())
