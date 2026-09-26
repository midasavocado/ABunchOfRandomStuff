#!/usr/bin/env bash
# Rebuild everything: voiceover → timing → sound effects/music → frames → MP4.
# Usage: tools/build.sh MODELS_DIR BUILD_DIR [FPS] [WORKERS]
#   MODELS_DIR holds kokoro-v1.0.onnx and voices-v1.0.bin (kokoro-onnx release "model-files-v1.0").
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS="$1"; BUILD="$2"; FPS="${3:-60}"; WORKERS="${4:-4}"
FFMPEG="${FFMPEG:-$(python3 -c 'import imageio_ffmpeg as i; print(i.get_ffmpeg_exe())')}"
export FFMPEG
mkdir -p "$BUILD"
python3 "$HERE/make_vo.py" --models "$MODELS" --out "$BUILD"          # also writes ../cues.js
node "$HERE/render.js" events "$BUILD/events.json"
python3 "$HERE/make_audio.py" --build "$BUILD"
node "$HERE/render.js" all "$BUILD/video.mp4" "$FPS" "$WORKERS"
"$FFMPEG" -y -loglevel error -i "$BUILD/video.mp4" -i "$BUILD/soundtrack.wav" -map 0:v -map 1:a \
  -c:v copy -c:a aac -b:a 256k -ar 48000 -shortest -movflags +faststart "$HERE/../macbook-showdown.mp4"
"$FFMPEG" -y -loglevel error -i "$BUILD/soundtrack.wav" -c:a libmp3lame -b:a 192k "$HERE/../assets/soundtrack.mp3"
echo "wrote macbook-showdown.mp4"
