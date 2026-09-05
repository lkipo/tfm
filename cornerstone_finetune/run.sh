#!/usr/bin/env bash
# Detached launcher for cornerstone_finetune/train.py.
#
# Survives the SSH session by three independent mechanisms, because any one of
# them alone has a hole:
#   1. setsid  -- the trainer leads its own session and process group, so the
#                 SIGHUP that follows a dropped connection is never delivered
#                 to it (nohup alone only masks SIGHUP for the shell's child;
#                 it does not detach the process group).
#   2. nohup + redirection -- stdout/stderr go to a file, so nothing blocks on
#                 a vanished pty and no output is lost.
#   3. auto-resume -- train.py restarts from checkpoints/last.ckpt, so even a
#                 host reboot costs at most one epoch plus whatever the 30 min
#                 periodic checkpoint has not covered. `start` after a crash is
#                 the resume command; there is no separate one.
#
#   ./cornerstone_finetune/run.sh prepare          # one-off data cache (~10 min)
#   ./cornerstone_finetune/run.sh smoke            # 2 epochs, 2 subjects
#   ./cornerstone_finetune/run.sh start            # core6, detached
#   ./cornerstone_finetune/run.sh start full11     # or any configs/*.yaml
#   ./cornerstone_finetune/run.sh status
#   ./cornerstone_finetune/run.sh logs 100
#   ./cornerstone_finetune/run.sh attach           # tail -f; Ctrl-C is safe
#   ./cornerstone_finetune/run.sh tb               # TensorBoard on :6010
#   ./cornerstone_finetune/run.sh stop             # SIGTERM, checkpoints first
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PYTHON="${PYTHON:-$REPO/old-LightningMedSeg3D/.venv/bin/python}"
TB_PORT="${TB_PORT:-6010}"
CONFIG_NAME="${2:-core6}"
CONFIG="$HERE/configs/${CONFIG_NAME%.yaml}.yaml"
LOG_DIR="$HERE/logs"
mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "python not found at $PYTHON -- set PYTHON=/path/to/python" >&2
  exit 2
fi

run_name() {
  "$PYTHON" - "$CONFIG" <<'PY' 2>/dev/null || basename "${CONFIG%.yaml}"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get("run_name") or sys.argv[1].rsplit("/", 1)[-1][:-5])
PY
}

RUN_NAME="$(run_name)"
RUN_DIR="$REPO/runs/cornerstone/$RUN_NAME"
PID_FILE="$LOG_DIR/$RUN_NAME.pid"
TB_PID_FILE="$LOG_DIR/tensorboard.pid"

alive() { [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }
latest_log() { ls -t "$LOG_DIR/${RUN_NAME}_"*.log 2>/dev/null | head -1; }

cmd_prepare() {
  echo "==> preparing the cached cohort (idempotent; --force to rebuild)"
  "$PYTHON" "$HERE/prepare_data.py" "${@:2}"
}

cmd_smoke() {
  echo "==> smoke test with $CONFIG (foreground)"
  "$PYTHON" -u "$HERE/train.py" --config "$CONFIG" --smoke --resume never
}

cmd_start() {
  if alive; then
    echo "already running as PID $(cat "$PID_FILE"); use 'status', or 'stop' first." >&2
    exit 1
  fi
  if [[ ! -f "$CONFIG" ]]; then
    echo "no such config: $CONFIG" >&2
    exit 2
  fi
  if [[ ! -f "$REPO/data/cornerstone_prepared/manifest.json" ]]; then
    echo "data cache missing -- run '$0 prepare' first." >&2
    exit 2
  fi
  local log="$LOG_DIR/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
  if [[ -f "$RUN_DIR/checkpoints/last.ckpt" ]]; then
    echo "==> resuming $RUN_NAME from checkpoints/last.ckpt"
  else
    echo "==> starting $RUN_NAME from scratch"
  fi
  CORNERSTONE_DETACHED=1 setsid nohup "$PYTHON" -u "$HERE/train.py" \
      --config "$CONFIG" --resume auto >>"$log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$PID_FILE"
  disown "$pid" 2>/dev/null || true
  sleep 2
  if kill -0 "$pid" 2>/dev/null; then
    echo "    PID    $pid  (also in $PID_FILE)"
    echo "    log    $log"
    echo "    run    $RUN_DIR"
    echo "    watch  $0 status   |   $0 attach   |   $0 tb"
  else
    echo "process exited immediately -- last lines of $log:" >&2
    tail -30 "$log" >&2
    rm -f "$PID_FILE"
    exit 1
  fi
}

cmd_status() {
  echo "run:     $RUN_NAME   ($CONFIG)"
  if alive; then
    local pid; pid="$(cat "$PID_FILE")"
    echo "state:   RUNNING as PID $pid"
    ps -o pid,etime,pcpu,pmem,rss --no-headers -p "$pid" 2>/dev/null \
      | awk '{printf "         uptime %s, cpu %s%%, rss %.1f GB\n", $2, $3, $5/1048576}'
  else
    if [[ -f "$PID_FILE" ]]; then
      echo "state:   not running (stale PID file cleared)"
      rm -f "$PID_FILE"
    else
      echo "state:   not running"
    fi
  fi
  if command -v nvidia-smi >/dev/null; then
    echo "gpu:     $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
                    --format=csv,noheader 2>/dev/null | head -1)"
  fi
  local log; log="$(latest_log)"
  if [[ -n "$log" ]]; then
    echo "log:     $log"
    echo "--- last epoch lines ---"
    grep -a "^\[epoch" "$log" | tail -3
    echo "--- tail ---"
    tail -3 "$log"
  fi
  if [[ -d "$RUN_DIR/checkpoints" ]]; then
    echo "--- checkpoints ---"
    ls -1sh "$RUN_DIR/checkpoints" 2>/dev/null | tail -8
  fi
  if [[ -f "$TB_PID_FILE" ]] && kill -0 "$(cat "$TB_PID_FILE")" 2>/dev/null; then
    echo "tb:      http://localhost:$TB_PORT (PID $(cat "$TB_PID_FILE"))"
  fi
}

cmd_logs()   { local log; log="$(latest_log)"; [[ -n "$log" ]] && tail -n "${2:-50}" "$log" || echo "no log yet"; }
cmd_attach() { local log; log="$(latest_log)"; [[ -n "$log" ]] && { echo "(Ctrl-C detaches; training keeps going)"; tail -f "$log"; } || echo "no log yet"; }

cmd_stop() {
  if ! alive; then echo "not running"; return 0; fi
  local pid; pid="$(cat "$PID_FILE")"
  echo "==> SIGTERM to $pid (writes checkpoints/interrupted.ckpt, then exits)"
  kill -TERM "$pid"
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then
    echo "    still alive after 60 s; use '$0 kill'"
  else
    echo "    stopped. '$0 start' resumes from last.ckpt."
    rm -f "$PID_FILE"
  fi
}

cmd_kill() {
  [[ -f "$PID_FILE" ]] && kill -9 -- "-$(cat "$PID_FILE")" 2>/dev/null
  # Dataloader workers are children of the trainer, but a SIGKILL'd parent
  # orphans them still holding GPU memory; the process-group kill above covers
  # them because setsid gave the run its own group.
  pkill -9 -f "cornerstone_finetune/train.py" 2>/dev/null
  rm -f "$PID_FILE"
  echo "killed."
}

cmd_tb() {
  if [[ -f "$TB_PID_FILE" ]] && kill -0 "$(cat "$TB_PID_FILE")" 2>/dev/null; then
    echo "already at http://localhost:$TB_PORT (PID $(cat "$TB_PID_FILE"))"; return 0
  fi
  local log="$LOG_DIR/tensorboard.log"
  setsid nohup "$PYTHON" -m tensorboard.main --logdir "$REPO/runs/cornerstone" \
      --port "$TB_PORT" --bind_all >>"$log" 2>&1 < /dev/null &
  echo $! > "$TB_PID_FILE"
  sleep 2
  echo "tensorboard on http://localhost:$TB_PORT (all runs under runs/cornerstone)"
  echo "over ssh:  ssh -N -L $TB_PORT:localhost:$TB_PORT $(whoami)@$(hostname)"
}

cmd_tb_stop() { [[ -f "$TB_PID_FILE" ]] && kill "$(cat "$TB_PID_FILE")" 2>/dev/null; rm -f "$TB_PID_FILE"; echo "tensorboard stopped"; }

case "${1:-status}" in
  prepare)  cmd_prepare "$@" ;;
  smoke)    cmd_smoke "$@" ;;
  start)    cmd_start "$@" ;;
  status)   cmd_status "$@" ;;
  logs)     cmd_logs "$@" ;;
  attach)   cmd_attach "$@" ;;
  stop)     cmd_stop "$@" ;;
  kill)     cmd_kill "$@" ;;
  tb)       cmd_tb "$@" ;;
  tb-stop)  cmd_tb_stop "$@" ;;
  *)        sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ; exit 1 ;;
esac
