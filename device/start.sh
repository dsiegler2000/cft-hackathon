#!/bin/bash
# Start llama-server (127.0.0.1:8081) and the web app (127.0.0.1:8080) if they are not running.
# Run from the UART console; laptop/serial_relay.py calls it on startup.
cd ~/lfm || exit 1

if ! curl -sf http://127.0.0.1:8081/health >/dev/null; then
  setsid nohup ./llama/llama-b12374/llama-server \
    -m models/LFM3-Omni-2B-Q4_K_M.gguf --mmproj models/mmproj-LFM3-Omni-2B-Q8_0.gguf \
    --host 127.0.0.1 --port 8081 -c 8192 -t 8 --reasoning off --no-webui \
    > llama-server.log 2>&1 < /dev/null &
fi
if ! curl -sf -o /dev/null http://127.0.0.1:8080/; then
  setsid nohup python3 app.py > app.log 2>&1 < /dev/null &
fi

for _ in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8081/health >/dev/null && curl -sf -o /dev/null http://127.0.0.1:8080/; then
    echo SERVICES_READY
    exit 0
  fi
  sleep 1
done
echo SERVICES_NOT_READY
exit 1
