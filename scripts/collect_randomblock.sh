#!/usr/bin/env bash
# Batch-collect RandomBlock SpaceMouse demos: each layout is recorded once per colour.
#
#   scripts/collect_randomblock.sh START END [FIRST_COLOR]
#
#   scripts/collect_randomblock.sh 1 50          # layouts 1..50 x red/green/blue = 150 demos
#   scripts/collect_randomblock.sh 18 34         # one sitting of a split batch
#   scripts/collect_randomblock.sh 12 50 blue    # resume: layout 12 starts at blue
#
# For every (layout, colour) it launches play.py --record, converts the recording,
# and checks the dataset actually grew. A demo that was not saved (missed grasp /
# 60 s timeout / crash) is re-recorded, so no layout ends up missing a colour; after
# 3 misses in a row it asks whether to retry, skip, or quit.
#
# Ctrl-C stops cleanly after the current step. The dataset save can never be
# interrupted half-way, and the exact command to resume is printed on exit.
set -u

START=${1:?usage: $0 START END [FIRST_COLOR]}
END=${2:?usage: $0 START END [FIRST_COLOR]}
FIRST_COLOR=${3:-red}
COLORS=(red green blue)

cd "$(dirname "$0")/.."

ROLLOUT=logs/rollouts/eval_RandomBlock_with_SpaceMousePilot
OUTPUT=logs/data/randomblock_demos.npy
META=$OUTPUT.tasks.json

case "$FIRST_COLOR" in red|green|blue) ;; *) echo "FIRST_COLOR must be red, green or blue"; exit 2 ;; esac

count() { python -c "import json;print(len(json.load(open('$META'))))" 2>/dev/null || echo 0; }
summary() {
    python - "$META" <<'EOF' 2>/dev/null
import json, sys, collections
m = json.load(open(sys.argv[1]))
c = collections.Counter(v.get("task", "? ? ? ?").split()[3] for v in m.values())
print(f"{len(m)} demos | red {c['red']}  green {c['green']}  blue {c['blue']}")
EOF
}

# Run a command immune to Ctrl-C. In a non-interactive shell, background jobs ignore
# SIGINT, while this script's own trap still records the stop request.
run_protected() {
    "$@" &
    local pid=$! rc
    while :; do
        wait "$pid"; rc=$?
        kill -0 "$pid" 2>/dev/null || break
    done
    return "$rc"
}

stop=0
trap 'stop=1; echo; echo ">>> stop requested - finishing the current step"' INT

resume_hint() {
    echo "Resume with:  scripts/collect_randomblock.sh $1 $END $2"
    echo "Dataset: $(summary)"
}

# ---- pre-flight -------------------------------------------------------------------
if command -v lsusb >/dev/null && ! lsusb -d 256f: >/dev/null 2>&1; then
    echo "SpaceMouse not found (lsusb -d 256f: is empty). Plug it in and re-run."; exit 1
fi
if pgrep -f "scripts/play.py" >/dev/null; then
    echo "Another play.py is running and would hold the SpaceMouse:"; pgrep -af "scripts/play.py"; exit 1
fi

if [ -f "$OUTPUT" ]; then
    backup=logs/data/backups/$(date +%Y%m%d_%H%M%S)
    mkdir -p "$backup" && cp "$OUTPUT" "$META" "$backup"/ && echo "Backed up dataset to $backup/"
fi
echo "Starting at: $(summary)"
echo "Layouts $START..$END, first colour $FIRST_COLOR"

# ---- collection loop --------------------------------------------------------------
started=0
for L in $(seq "$START" "$END"); do
    for C in "${COLORS[@]}"; do
        if [ "$started" = 0 ]; then
            [ "$C" = "$FIRST_COLOR" ] || continue
            started=1
        fi

        misses=0
        while :; do
            if [ "$stop" = 1 ]; then resume_hint "$L" "$C"; exit 130; fi

            echo
            echo "================  layout $L / $C   (attempt $((misses + 1)))  ================"
            before=$(count)

            python scripts/play.py --task RandomBlock --pilot SpaceMousePilot \
                --num_envs 1 --record --yes --layout_seed "$L" --target_color "$C"

            # Convert even after a Ctrl-C: it keeps only successful episodes, so this
            # can only save a finished demo, never add a broken one.
            if [ -d "$ROLLOUT" ]; then
                run_protected python scripts/convert_demos.py \
                    --rollout_dir "$ROLLOUT" --output "$OUTPUT" --append
            fi

            after=$(count)
            if [ "$after" -gt "$before" ]; then
                echo ">>> saved layout $L / $C  ->  $(summary)"
                break
            fi

            if [ "$stop" = 1 ]; then resume_hint "$L" "$C"; exit 130; fi

            misses=$((misses + 1))
            echo ">>> layout $L / $C was NOT saved (not successful, timed out, or crashed)"
            if [ "$misses" -ge 3 ]; then
                read -r -p ">>> $misses misses in a row. [r]etry, [s]kip this colour, [q]uit? " ans </dev/tty || ans=q
                [ "$stop" = 1 ] && ans=q
                case "$ans" in
                    s|S) echo ">>> skipping layout $L / $C"; break ;;
                    q|Q) resume_hint "$L" "$C"; exit 1 ;;
                    *)   misses=0 ;;
                esac
            else
                echo ">>> re-recording it"
            fi
        done
    done
done

echo
echo "Done: layouts $START..$END."
echo "Dataset: $(summary)"
