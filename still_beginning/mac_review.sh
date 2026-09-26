#!/bin/bash
# One-shot review pass for the Mac:  bash still_beginning/mac_review.sh
#  1. pulls the latest scenes, 2. renders the whole film as a 960x540 preview (every shot fresh, with the score),
#  3. puts the film + per-shot contact sheets in ~/Desktop/SB_review, 4. pushes a small review copy back to the repo
#     (review/) so the cloud session can inspect every cut.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OUT="$HOME/Desktop/SB_review"
cd "$REPO"
git checkout still_beginning
git pull --ff-only || true
git lfs pull
cd "$HERE"
if [ "$1" = "--reuse" ]; then
  # shots already rendered: just re-assemble the film
  python3 render_all.py --preview --master-only
else
  # never reuse stale preview shots
  rm -f edit/*_prev.mov STILL_BEGINNING.mp4
  python3 render_all.py --preview
fi
mkdir -p "$OUT" review
# small review copy (960x540, same cut + score)
ffmpeg -y -loglevel error -i STILL_BEGINNING.mp4 -vf scale=960:540 -c:v libx264 -crf 20 -preset medium \
  -c:a aac -b:a 192k -movflags +faststart review/preview.mp4
cp review/preview.mp4 "$OUT/STILL_BEGINNING_preview.mp4"
# contact sheets: first / middle / last frame of every shot, in edit order
python3 - "$OUT" <<'PY'
import sys, os, subprocess
sys.path.insert(0, "lib")
import timeline as TL
out = sys.argv[1]
os.makedirs("review/frames", exist_ok=True)
for sid, a, b, sc in TL.SHOTS:
    for tag, f in (("a", a), ("m", (a + b) // 2), ("z", b - 1)):
        p = "review/frames/%s_%s.jpg" % (sid, tag)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "review/preview.mp4", "-vf",
                        "select=eq(n\\,%d),scale=480:270" % f, "-frames:v", "1", "-q:v", "3", p], check=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "1", "-pattern_type", "glob", "-i", "review/frames/*.jpg",
                "-vf", "tile=6x10", "-frames:v", "2", "-q:v", "3", os.path.join(out, "sheet_%d.jpg")], check=True)
print("sheets written to", out)
PY
cp -R review/frames "$OUT/"
cd "$REPO"
git add still_beginning/review
git commit -m "review: Mac preview of the full film + per-shot frames" || true
git push
echo
echo "Done. Film + sheets: $OUT   (review copy pushed to the repo)"
open "$OUT"
