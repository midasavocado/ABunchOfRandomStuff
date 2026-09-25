import sys, os, glob
from PIL import Image, ImageDraw
d = sys.argv[1]; out = sys.argv[2]; step = int(sys.argv[3]) if len(sys.argv) > 3 else 10; cols = int(sys.argv[4]) if len(sys.argv) > 4 else 3
fs = sorted(glob.glob(os.path.join(d, "*.png")))[::step]
w, h = 480, 270
rows = (len(fs) + cols - 1) // cols
sheet = Image.new("RGB", (w * cols, h * rows))
for i, f in enumerate(fs):
    im = Image.open(f).convert("RGB").resize((w, h))
    ImageDraw.Draw(im).text((6, 4), os.path.basename(f)[:-4], fill=(255, 255, 0))
    sheet.paste(im, ((i % cols) * w, (i // cols) * h))
sheet.save(out)
