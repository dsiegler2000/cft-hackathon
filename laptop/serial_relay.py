#!/usr/bin/env python3
"""Laptop side of the serial relay: http://localhost:8080 -> UART -> the board's app.

The guest Wi-Fi isolates clients, so the laptop cannot reach the board over the network.
Each HTTP request is typed into the logged-in shell on the board's UART console, replayed
there by ~/lfm/serial_fwd.py, and the response comes back the same way.

On startup the relay moves the console from 115200 to 921600 baud (about 90 KB/s, intact
in both directions on this board; 3 Mbaud corrupts uploads) and runs ~/lfm/start.sh so
llama-server and the app are up. To use the console by hand afterwards:
    screen /dev/cu.usbserial-NNPMQ11I000K1 921600
A reboot puts the console back at 115200.

Usage: python3 serial_relay.py [--port /dev/cu.usbserial-...K1] [--listen 8080]
Needs a logged-in shell on the console and nothing else (screen) holding the port.
"""
import argparse, base64, fcntl, hashlib, json, os, re, struct, sys, termios, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = "/dev/cu.usbserial-NNPMQ11I000K1"
FAST, SLOW = 921600, 115200
IOSSIOSPEED = 0x80085402  # macOS: set any baud rate on a serial port
PAD = b" " * 16384         # pushes the tail of an upload through the board's UART
HOP_BY_HOP = {"connection", "transfer-encoding", "keep-alive", "content-length"}
KILL_LINE = b"\x15"        # Ctrl-U: clears stray padding from the shell's input line


class Console:
    """The board's UART console; runs one command at a time."""

    def __init__(self, path):
        self.lock = threading.Lock()
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        a = termios.tcgetattr(self.fd)
        a[0] = a[1] = a[3] = 0
        a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        a[4] = a[5] = SLOW
        termios.tcsetattr(self.fd, termios.TCSANOW, a)

    def speed(self, baud):
        fcntl.ioctl(self.fd, IOSSIOSPEED, struct.pack("Q", baud))
        termios.tcflush(self.fd, termios.TCIOFLUSH)

    def write(self, data):
        while data:
            try:
                data = data[os.write(self.fd, data):]
            except BlockingIOError:
                time.sleep(0.002)

    def read_until(self, pattern, timeout):
        buf, end, pat = b"", time.time() + timeout, re.compile(pattern)
        while time.time() < end:
            try:
                chunk = os.read(self.fd, 262144)
            except BlockingIOError:
                time.sleep(0.003)
                continue
            buf += chunk
            if pat.search(buf[-len(chunk) - 256:]):
                return buf
        raise TimeoutError(f"no reply from the board console (last output: {buf[-120:]!r})")

    def ping(self):
        try:
            self.write(KILL_LINE + b"\necho P_$((6*7))_P\n")
            return b"P_42_P" in self.read_until(rb"P_42_P|login: $", 2.5)
        except TimeoutError:
            return False

    def connect(self):
        """Find the console's speed and move it to FAST."""
        self.speed(FAST)
        if self.ping():
            return
        self.speed(SLOW)
        if not self.ping():
            self.write(b"\n")
            try:
                if b"login:" in self.read_until(rb"login: $", 2):
                    sys.exit("The board console is at a login prompt. Log in with screen, quit screen, then rerun.")
            except TimeoutError:
                pass
            sys.exit("No shell answered on the board console at 921600 or 115200 baud.")
        self.write(b"stty -F /dev/ttyMSM0 %d\n" % FAST)
        time.sleep(1)
        self.speed(FAST)
        if not self.ping():
            self.speed(SLOW)
            sys.exit("The console did not answer after switching to 921600 baud.")

    def run(self, cmd, timeout=60):
        """Run a short shell command, return its output."""
        with self.lock:
            tag = uuid.uuid4().hex[:8]
            self.write(KILL_LINE + f"echo __S{tag}__; {cmd}; echo __E{tag}_$?__\n".encode())
            text = self.read_until(rb"__E" + tag.encode() + rb"_\d+__", timeout).decode(errors="replace")
            return text[text.rfind(f"__S{tag}__") + len(tag) + 5:text.rfind(f"__E{tag}_")]

    def http(self, method, path, headers, body, timeout=600):
        """Replay one HTTP request on the board through serial_fwd.py."""
        with self.lock:
            tag = uuid.uuid4().hex[:8]
            meta = base64.b64encode(json.dumps({"method": method, "path": path, "headers": headers}).encode())
            b64 = base64.encodebytes(body) if body else b""
            self.write(KILL_LINE + b"python3 ~/lfm/serial_fwd.py %s %s %d %s; echo __E%s_$?__\n" % (
                tag.encode(), meta, len(b64), hashlib.sha256(body).hexdigest().encode(), tag.encode()))
            self.read_until(b"READY_" + tag.encode(), 10)
            if b64:
                self.write(b64 + PAD)
            out = self.read_until(rb"__E" + tag.encode() + rb"_\d+__", timeout).decode(errors="replace")
        out = out.replace("\r", "")
        reply = out[out.find(f"__R{tag}__"):].split("\n")
        head = json.loads(base64.b64decode(next(l[2:] for l in reply if l.startswith("H "))))
        want = next(l[2:].strip() for l in reply if l.startswith("Z "))
        data = base64.b64decode("".join(l for l in reply[2:] if re.fullmatch(r"[A-Za-z0-9+/=]+", l)))
        if hashlib.sha256(data).hexdigest() != want:
            raise IOError("response corrupted on the serial link")
        return head["status"], head["headers"], data


def make_handler(console):
    class Relay(BaseHTTPRequestHandler):
        def _relay(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n) if n else b""
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
            for attempt in range(2):
                try:
                    status, resp_headers, data = console.http(self.command, self.path, headers, body)
                    if status != 599 or attempt:  # 599: upload corrupted, retry once
                        break
                except Exception as e:
                    status, resp_headers = 502, [["Content-Type", "application/json"]]
                    data = json.dumps({"error": f"serial relay: {type(e).__name__}: {e}"}).encode()
            self.send_response(status)
            for k, v in resp_headers:
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = _relay

        def log_message(self, fmt, *args):
            sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    return Relay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--listen", type=int, default=8080)
    args = ap.parse_args()
    console = Console(args.port)
    console.connect()
    console.run("stty -echo cols 250", timeout=10)
    print("Console at 921600 baud. Starting services on the board...", flush=True)
    print(console.run("bash ~/lfm/start.sh", timeout=180).strip(), flush=True)
    print(f"Relay ready: http://localhost:{args.listen}  (via {args.port})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.listen), make_handler(console)).serve_forever()


if __name__ == "__main__":
    main()
