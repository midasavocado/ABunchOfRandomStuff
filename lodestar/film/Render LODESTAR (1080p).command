#!/bin/bash
# LODESTAR — double-click to render the film on this Mac's GPU (full film at 1920x1080 — recommended).
cd "$(dirname "$0")" || exit 1
clear
echo
echo "   L  O  D  E  S  T  A  R"
echo "   the film renderer  —  preset: 1080p"
echo
echo "   Every frame is ray-traced on your GPU from shaders — nothing is pre-made."
echo "   You can leave it running. If you stop it, double-click again and it resumes."
echo
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "   First run: installing 'uv' (a small self-contained Python manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "   Could not install uv — check the internet connection."; read -r -p "   Press Enter to close."; exit 1; }
  export PATH="$HOME/.local/bin:$PATH"
fi
if uv run --quiet --python 3.12 render.py --preset 1080p; then
  echo
  echo "   Done! Opening LODESTAR_1080p.mp4 ..."
  open "LODESTAR_1080p.mp4"
else
  echo
  echo "   Something went wrong (see the messages above)."
fi
echo
read -r -p "   Press Enter to close this window."
