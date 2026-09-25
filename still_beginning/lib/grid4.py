import sys
from PIL import Image, ImageDraw
out = sys.argv[1]; names = ["TOP", "THUMB SIDE", "FRONT", "PERSP"]
S = Image.new("RGB", (1200, 1200))
for i in range(4):
    im = Image.open(f"/tmp/_o{i}.png").convert("RGB"); ImageDraw.Draw(im).text((8, 8), names[i], fill=(255, 255, 0))
    S.paste(im, ((i % 2) * 600, (i // 2) * 600))
S.save(out)
