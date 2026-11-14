#!/usr/bin/env python3
"""Laptop side of the serial relay: http://localhost:8080 -> UART -> the board's app.

The guest Wi-Fi isolates clients, so the laptop cannot reach the board over the network.
Each HTTP request is typed into the logged-in shell on the board's UART console, replayed
there by ~/lfm/serial_fwd.py, and the response comes back the same way.

On startup the relay moves the console from 115200 to 921600 baud (about 90 KB/s, intact
in both directions on this board; 3 Mbaud corrupts uploads) and runs ~/lfm/start.sh so
llama-server and the app are up. A reboot puts the console back at 115200.

Stability:
- The port is opened exclusively (TIOCEXCL): any other program that opens it gets
  "Resource busy" instead of interleaving its bytes with ours.
- After a timeout or a garbled reply the console is resynced (interrupt, clear the line,
  re-detect the speed) and the request is retried once. A vanished USB port is reopened.
- An idle health check pings the console every 30 s and resyncs it when needed.
Run it under laptop/run_relay.sh to restart it if it ever exits.

To use the console by hand, stop the relay first, then:
    screen /dev/cu.usbserial-NNPMQ11I000K1 921600

Usage: python3 serial_relay.py [--port /dev/cu.usbserial-...K1] [--listen 8080]
Needs a logged-in shell on the console.
"""
import argparse, base64, fcntl, hashlib, json, os, re, signal, struct, sys, termios, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = "/dev/cu.usbserial-NNPMQ11I000K1"
FAST, SLOW = 921600, 115200
IOSSIOSPEED = 0x80085402  # macOS: set any baud rate on a serial port
TIOCEXCL = getattr(termios, "TIOCEXCL", 0x2000740D)
TIOCOUTQ = 0x40047473      # macOS: bytes waiting in the output queue
MAX_QUEUED = 16384
PAD = b" " * 16384         # pushes the tail of an upload through the board's UART
HOP_BY_HOP = {"connection", "transfer-encoding", "keep-alive", "content-length"}
KILL_LINE = b"\x15"        # Ctrl-U: clears stray padding from the shell's input line
HEALTH_EVERY = 30


class ConsoleError(Exception):
    """The console is unusable until the user acts (login prompt, no shell)."""


class StalledError(ConsoleError):
    """The Mac's USB-serial channel stopped sending; only re-plugging the cable clears it."""


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Console:
    """The board's UART console; runs one command at a time."""

    def __init__(self, path):
        self.path, self.fd = path, None
        self.lock = threading.RLock()
        self.last_used = time.time()
        self.open()

    # ---- port ---------------------------------------------------------------------

    def open(self):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
        try:
            self.fd = os.open(self.path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        except OSError as e:
            self.fd = None
            raise ConsoleError(f"cannot open {self.path}: {e.strerror}. "
                               f"Check `lsof {self.path}` for another program holding it.")
        fcntl.ioctl(self.fd, TIOCEXCL)
        a = termios.tcgetattr(self.fd)
        a[0] = a[1] = a[3] = 0
        a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        a[4] = a[5] = SLOW
        termios.tcsetattr(self.fd, termios.TCSANOW, a)

    def speed(self, baud):
        fcntl.ioctl(self.fd, IOSSIOSPEED, struct.pack("Q", baud))
        termios.tcflush(self.fd, termios.TCIOFLUSH)

    def out_queue(self):
        return struct.unpack("i", fcntl.ioctl(self.fd, TIOCOUTQ, struct.pack("i", 0)))[0]

    def write(self, data, timeout=30):
        """Write in small steps, keeping at most MAX_QUEUED bytes in the driver: if this
        process dies with a full output queue, the macOS FTDI driver can jam the channel."""
        end = time.time() + timeout
        while data:
            if self.out_queue() < MAX_QUEUED:
                try:
                    data = data[os.write(self.fd, data[:4096]):]
                    continue
                except BlockingIOError:
                    pass
            if time.time() > end:
                termios.tcflush(self.fd, termios.TCOFLUSH)
                raise StalledError("serial output stalled: the USB-serial channel is jammed. "
                                   "Unplug and re-plug the board's micro-USB debug cable")
            time.sleep(0.002)

    def close(self):
        """Drop anything still queued and release the port."""
        if self.fd is not None:
            termios.tcflush(self.fd, termios.TCIOFLUSH)
            os.close(self.fd)
            self.fd = None

    def drain(self, secs):
        end, out = time.time() + secs, b""
        while time.time() < end:
            try:
                out += os.read(self.fd, 262144)
            except BlockingIOError:
                time.sleep(0.01)
        return out

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

    # ---- shell state ----------------------------------------------------------------

    def ping(self, timeout=2.5):
        try:
            self.write(KILL_LINE + b"\necho P_$((6*7))_P\n")
            return b"P_42_P" in self.read_until(rb"P_42_P", timeout)
        except TimeoutError:
            return False

    def at_login_prompt(self):
        self.write(b"\n")
        return b"login:" in self.drain(1.5)

    def connect(self):
        """Get a responsive shell at FAST baud, recovering from a stuck console."""
        with self.lock:
            for attempt in range(3):
                for baud in (FAST, SLOW):
                    self.speed(baud)
                    if self.ping():
                        if baud == SLOW:
                            self.write(b"stty -F /dev/ttyMSM0 %d\n" % FAST)
                            time.sleep(1)
                            self.speed(FAST)
                            if not self.ping():
                                continue
                        self.write(KILL_LINE + b"stty -echo cols 250\n")
                        self.drain(0.5)
                        return
                    if self.at_login_prompt():
                        raise ConsoleError("the board console is at a login prompt: log in with "
                                           "`screen` (at 115200 or 921600 baud), quit screen, and the relay retries")
                self.unstick()
            raise ConsoleError("no shell answered on the board console at 921600 or 115200 baud")

    def unstick(self):
        """Free a console left mid-command: satisfy a raw-mode reader, interrupt, reset."""
        for baud in (FAST, SLOW):
            self.speed(baud)
            self.write(PAD)              # a serial_fwd.py waiting in raw mode gets its bytes
            time.sleep(0.3)
            self.write(b"\x03\x03\x15")  # interrupt whatever runs, clear the line
            self.drain(0.5)
            self.write(b"\nstty sane; stty -echo\n")
            self.drain(0.5)

    def recover(self):
        log("console recovery...")
        try:
            os.fstat(self.fd)
            if not os.path.exists(self.path):
                raise OSError("port vanished")
        except OSError:
            self.open()
        self.connect()
        log("console recovered")

    # ---- commands -------------------------------------------------------------------

    def run(self, cmd, timeout=60):
        """Run a short shell command, return its output."""
        with self.lock:
            self.last_used = time.time()
            tag = uuid.uuid4().hex[:8]
            self.write(KILL_LINE + f"echo __S{tag}__; {cmd}; echo __E{tag}_$?__\n".encode())
            text = self.read_until(rb"__E" + tag.encode() + rb"_\d+__", timeout).decode(errors="replace")
            return text[text.rfind(f"__S{tag}__") + len(tag) + 5:text.rfind(f"__E{tag}_")]

    def http(self, method, path, headers, body, timeout=600):
        """Replay one HTTP request on the board through serial_fwd.py."""
        with self.lock:
            self.last_used = time.time()
            tag = uuid.uuid4().hex[:8]
            meta = base64.b64encode(json.dumps({"method": method, "path": path, "headers": headers}).encode())
            b64 = base64.encodebytes(body) if body else b""
            self.write(KILL_LINE + b"python3 ~/lfm/serial_fwd.py %s %s %d %s; echo __E%s_$?__\n" % (
                tag.encode(), meta, len(b64), hashlib.sha256(body).hexdigest().encode(), tag.encode()))
            self.read_until(b"READY_" + tag.encode(), 10)
            if b64:
                self.write(b64 + PAD)
            out = self.read_until(rb"__E" + tag.encode() + rb"_\d+__", timeout).decode(errors="replace")
            self.last_used = time.time()
        out = out.replace("\r", "")
        reply = out[out.find(f"__R{tag}__"):].split("\n")
        head = json.loads(base64.b64decode(next(l[2:] for l in reply if l.startswith("H "))))
        want = next(l[2:].strip() for l in reply if l.startswith("Z "))
        data = base64.b64decode("".join(l for l in reply[2:] if re.fullmatch(r"[A-Za-z0-9+/=]+", l)))
        if hashlib.sha256(data).hexdigest() != want:
            raise IOError("response corrupted on the serial link")
        return head["status"], head["headers"], data

    def health_loop(self):
        """Ping the console when idle; recover it if it stopped answering."""
        while True:
            time.sleep(5)
            if time.time() - self.last_used < HEALTH_EVERY or not self.lock.acquire(blocking=False):
                continue
            try:
                self.last_used = time.time()
                if not self.ping(timeout=4):
                    self.recover()
            except ConsoleError as e:
                log(f"health check: {e}")
            except Exception as e:
                log(f"health check failed: {type(e).__name__}: {e}")
            finally:
                self.lock.release()


def make_handler(console):
    class Relay(BaseHTTPRequestHandler):
        def _relay(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n) if n else b""
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
            for attempt in range(2):
                try:
                    status, resp_headers, data = console.http(self.command, self.path, headers, body)
                    if status != 599:  # 599: upload corrupted on the link
                        break
                    error = data.decode(errors="replace")
                except ConsoleError as e:
                    status, error = 503, str(e)
                    break
                except Exception as e:
                    status, error = 502, f"{type(e).__name__}: {e}"
                log(f"{self.command} {self.path} attempt {attempt + 1} failed: {error}")
                try:
                    console.recover()
                except Exception as e:
                    status, error = 503, f"{type(e).__name__}: {e}"
                    break
            if status in (502, 503, 599):
                resp_headers = [["Content-Type", "application/json"]]
                data = json.dumps({"error": f"serial relay: {error}"}).encode()
            self.send_response(status)
            for k, v in resp_headers:
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = _relay

        def log_message(self, fmt, *args):
            log(fmt % args)

    return Relay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--listen", type=int, default=8080)
    args = ap.parse_args()
    console = None

    def shutdown(signum, frame):
        if console:
            console.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        console = Console(args.port)
        console.connect()
        log("Console at 921600 baud. Starting services on the board...")
        log(console.run("bash ~/lfm/start.sh", timeout=180).strip())
    except (ConsoleError, TimeoutError) as e:
        if console:
            console.close()
        sys.exit(f"relay not started: {e}")
    threading.Thread(target=console.health_loop, daemon=True).start()
    log(f"Relay ready: http://localhost:{args.listen}  (via {args.port}, exclusive)")
    ThreadingHTTPServer(("127.0.0.1", args.listen), make_handler(console)).serve_forever()


if __name__ == "__main__":
    main()
