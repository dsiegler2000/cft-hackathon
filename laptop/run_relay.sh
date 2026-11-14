#!/bin/bash
# Keep the serial relay running: restart it whenever it exits.
#   laptop/run_relay.sh            run in the foreground (Ctrl-C to stop)
#   laptop/run_relay.sh --detach   run in the background, log to laptop/relay.log
#   laptop/run_relay.sh --stop     stop a detached relay
DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$DIR/relay.pid"

if [ "$1" = "--stop" ]; then
  [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null && echo "stopped"
  pkill -f "$DIR/serial_relay.py" 2>/dev/null
  rm -f "$PIDFILE"
  exit 0
fi
if [ "$1" = "--detach" ]; then
  # A new session, so closing the terminal (or the tool that started it) does not stop it.
  python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
    "$0" < /dev/null >> "$DIR/relay.log" 2>&1 &
  echo "relay running in the background (pid $!), log: $DIR/relay.log"
  exit 0
fi

echo $$ > "$PIDFILE"
trap 'pkill -P $$; rm -f "$PIDFILE"; exit 0' INT TERM
while true; do
  python3 -I "$DIR/serial_relay.py" &
  wait $!
  echo "$(date +%H:%M:%S) relay exited ($?), restarting in 10 s"
  sleep 10
done
