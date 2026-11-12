#!/bin/bash
# Runs the verification and the full benchmark on the GPU machine with the two
# services that share the GPU (Jellyfin and the ErsatzTV container) paused.
# They are restarted when the script ends, also if it fails or is interrupted.
#   bash scripts/run_on_gpu_host.sh [extra arguments for run_benchmarks.py]
set -u
cd "$(dirname "$0")/.."
OUT=../results/media_vm
mkdir -p "$OUT"
LOG="$OUT/service_log.txt"
restart() {
    sudo systemctl start jellyfin
    sudo docker start ersatztv > /dev/null
    echo "$(date -Is) services restarted: jellyfin=$(systemctl is-active jellyfin) ersatztv=$(sudo docker inspect -f '{{.State.Status}}' ersatztv)" >> "$LOG"
}
trap restart EXIT
sudo systemctl stop jellyfin
sudo docker stop ersatztv > /dev/null
echo "$(date -Is) services stopped: jellyfin=$(systemctl is-active jellyfin) ersatztv=$(sudo docker inspect -f '{{.State.Status}}' ersatztv)" >> "$LOG"
sleep 5
top -bn1 | head -15 > "$OUT/cpu_load_before.txt"
nvidia-smi > "$OUT/gpu_before.txt"
PY=../.venv/bin/python
if [ "${SKIP_VERIFY:-0}" != "1" ]; then
    $PY python/verify.py --pkt ../data/mawi_60s.pkt > "$OUT/verify_output.txt" 2>&1
    echo "$(date -Is) verify exit code $?" >> "$LOG"
    tail -1 "$OUT/verify_output.txt"
fi
$PY python/run_benchmarks.py --config configs/media_vm.json --out "$OUT" "$@" > "$OUT/benchmark_log.txt" 2>&1
echo "$(date -Is) benchmarks exit code $?" >> "$LOG"
tail -2 "$OUT/benchmark_log.txt"
