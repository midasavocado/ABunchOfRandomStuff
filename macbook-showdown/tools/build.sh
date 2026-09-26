#!/usr/bin/env bash
# Rebuild everything: voiceover → timing → soundtrack → frames → MP4.
# Usage: tools/build.sh MODELS_DIR SAMPLES_DIR DRUMS_DIR BUILD_DIR [FPS] [WORKERS]
#   MODELS_DIR   kokoro-v1.0.onnx and voices-v1.0.bin (kokoro-onnx release "model-files-v1.0")
#   SAMPLES_DIR  MusyngKite/ from github.com/gleitz/midi-js-soundfonts
#   DRUMS_DIR    drum-samples/ from github.com/Tonejs/audio
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS="$1"; SAMPLES="$2"; DRUMS="$3"; BUILD="$4"; FPS="${5:-60}"; WORKERS="${6:-4}"
FFMPEG="${FFMPEG:-$(python3 -c 'import imageio_ffmpeg as i; print(i.get_ffmpeg_exe())')}"
export FFMPEG
mkdir -p "$BUILD"
(cd "$HERE/.." && npm install --no-audit --no-fund && npm run bundle)
python3 "$HERE/make_vo.py" --models "$MODELS" --out "$BUILD"          # also writes ../cues.js
node "$HERE/render.js" events "$BUILD/events.json"
python3 "$HERE/make_audio.py" --build "$BUILD" --samples "$SAMPLES" --drums "$DRUMS"
node "$HERE/render.js" all "$BUILD/video.mp4" "$FPS" "$WORKERS"
"$FFMPEG" -y -loglevel error -i "$BUILD/video.mp4" -i "$BUILD/soundtrack.wav" -map 0:v -map 1:a \
  -c:v copy -c:a aac -b:a 256k -ar 48000 -shortest -movflags +faststart "$HERE/../macbook-showdown.mp4"
"$FFMPEG" -y -loglevel error -i "$BUILD/soundtrack.wav" -c:a libmp3lame -b:a 192k "$HERE/../assets/soundtrack.mp3"
echo "wrote macbook-showdown.mp4"
