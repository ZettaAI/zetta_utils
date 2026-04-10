#!/bin/bash
# Launch zetta run for (dataset, variant, agg_level) combinations.
# - Per-dataset variant/agg lists (use "none" variant for datasets with a single affinity model)
# - Each combo gets its own temp cue file (no edit races between parallel jobs)
# - Temp files are cleaned up at the end
#
# Usage:
#   bash scripts/run_seg_contacts_mouse_parallel.sh [dataset ...]
# Examples:
#   bash scripts/run_seg_contacts_mouse_parallel.sh
#   bash scripts/run_seg_contacts_mouse_parallel.sh jrc_mus_dorsal_striatum_2
#   bash scripts/run_seg_contacts_mouse_parallel.sh ac3 mouse_golden
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

declare -A SPECS=(
  [ac3]=specs/martin/contact_analysis/generate_seg_contacts_ac3.cue
  [mouse_golden]=specs/martin/contact_analysis/generate_seg_contacts_mouse_golden.cue
  [pinky]=specs/martin/contact_analysis/generate_seg_contacts_pinky.cue
  [jrc_mus_dorsal_striatum_2]=specs/martin/contact_analysis/generate_seg_contacts_jrc_mus-dorsal-striatum-2.cue
)

# Per-dataset variants. Use "none" for datasets without a #AFF_MODEL_VARIANT dimension.
declare -A VARIANTS_PER_DS=(
  [ac3]="16nm-overseg-no-imgdeg 16nm-no-overseg-no-imgdeg"
  [mouse_golden]="16nm-overseg-no-imgdeg 16nm-no-overseg-no-imgdeg"
  [pinky]="16nm-overseg-no-imgdeg 16nm-no-overseg-no-imgdeg"
  [jrc_mus_dorsal_striatum_2]="none"
)

declare -A AGG_LEVELS_PER_DS=(
  [ac3]="0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6"
  [mouse_golden]="0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6"
  [pinky]="0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6"
  [jrc_mus_dorsal_striatum_2]="0.3 0.35 0.4 0.45 0.5 0.55 0.6" # 0.6 exists already
)

MAX_PARALLEL=4

if [ "$#" -gt 0 ]; then
  DATASETS=("$@")
else
  DATASETS=(ac3 mouse_golden pinky)
fi

for ds in "${DATASETS[@]}"; do
  if [ -z "${SPECS[$ds]+x}" ]; then
    echo "Unknown dataset: $ds" >&2
    echo "Known datasets: ${!SPECS[*]}" >&2
    exit 1
  fi
done

TMPS=()
for ds in "${DATASETS[@]}"; do
  src="${SPECS[$ds]}"
  read -ra variants <<< "${VARIANTS_PER_DS[$ds]}"
  read -ra agg_levels <<< "${AGG_LEVELS_PER_DS[$ds]}"
  for variant in "${variants[@]}"; do
    for agg in "${agg_levels[@]}"; do
      if [ "$variant" = "none" ]; then
        tmp="$(dirname "$src")/_tmp_${ds}_agg${agg}.cue"
      else
        tmp="$(dirname "$src")/_tmp_${ds}_${variant}_agg${agg}.cue"
      fi
      cp "$src" "$tmp"
      if [ "$variant" != "none" ]; then
        sed -i "s|^#AFF_MODEL_VARIANT: \".*\"|#AFF_MODEL_VARIANT: \"$variant\"|" "$tmp"
      fi
      sed -i "s|^#AGG_LEVEL: \"[^\"]*\"|#AGG_LEVEL: \"$agg\"|" "$tmp"
      TMPS+=("$tmp")
    done
  done
done
echo "Generated ${#TMPS[@]} temp specs"

LOG_DIR="$REPO_ROOT/scripts/logs/seg_contacts_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Per-job logs will be written to: $LOG_DIR"

cleanup() {
  echo "Cleaning up temp specs..."
  for tmp in "${TMPS[@]}"; do rm -f "$tmp"; done
}
trap cleanup EXIT

for tmp in "${TMPS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    wait -n
  done
  job_name="$(basename "$tmp" .cue)"
  log_file="$LOG_DIR/${job_name}.log"
  echo "=== $(date '+%H:%M:%S') Launching $job_name (log: $log_file) ==="
  ( zetta run "$tmp" >"$log_file" 2>&1; echo "EXIT=$?" >>"$log_file" ) &
done

wait
echo "ALL DONE - logs in $LOG_DIR"
echo "To check for failures:"
echo "  grep -L '^EXIT=0' $LOG_DIR/*.log"
