"""Quick review without rendering the film: first / middle / last frame of every shot at 960x540, in edit order.
    python3 review_stills.py                 # all shots
    python3 review_stills.py s09b s15        # just these (re-render)
    python3 review_stills.py -j6             # 6 shots at once (default 4); finished shots are kept (resume)
Writes review/frames/<sid>_{a,m,z}.jpg + ~/Desktop/SB_review/sheet_N.jpg (one row per shot), then pushes review/."""
import os, sys, time, shutil, subprocess
import numpy as np
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import timeline as TL

ROOT = os.path.dirname(os.path.abspath(__file__))
BLENDER = os.environ.get("BLENDER") or shutil.which("blender") or "/Applications/Blender.app/Contents/MacOS/Blender"
OUT = os.path.expanduser("~/Desktop/SB_review")
FR = os.path.join(ROOT, "review", "frames")
only = [a for a in sys.argv[1:] if not a.startswith("-")]
os.makedirs(OUT, exist_ok=True); os.makedirs(FR, exist_ok=True)
shots = [s for s in TL.SHOTS if not only or s[0] in only]
JOBS = 4
for a_ in sys.argv[1:]:
    if a_.startswith("-j"):
        JOBS = int(a_[2:] or 4)
only = [a_ for a_ in only if not a_.startswith("-j")]


def done(sid, a, b):
    return all(os.path.exists(os.path.join(FR, "%s_%s.jpg" % (sid, t))) for t in "amz")


def work(item):
    i, (sid, a, b, scene) = item
    d = os.path.join(ROOT, "preview", sid)
    os.makedirs(d, exist_ok=True)
    for fn in os.listdir(d):
        if fn.endswith(".png"):
            os.remove(os.path.join(d, fn))
    t = time.time()
    env = dict(os.environ, SB_RES="preview", SB_FRAMES="first,mid,last", SB_SKIP_EXISTING="0")
    r = subprocess.run([BLENDER, "-b", "-P", os.path.join("scenes", scene + ".py"), "--", sid], cwd=ROOT, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    for tag, f in (("a", a), ("m", (a + b) // 2), ("z", b - 1)):
        p = os.path.join(d, "%04d.png" % f)
        if os.path.exists(p):
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if im.dtype != np.uint8:
                im = (im / 257).astype(np.uint8)
            cv2.imwrite(os.path.join(FR, "%s_%s.jpg" % (sid, tag)), im[..., :3], [cv2.IMWRITE_JPEG_QUALITY, 90])
    msg = "[%d/%d] %s %s %.0fs" % (i + 1, len(shots), sid, "ok" if r.returncode == 0 else "FAILED", time.time() - t)
    if r.returncode != 0:
        msg += "\n" + r.stderr[-1500:]
    return msg


t_all = time.time()
shots = [s for s in shots if not only or s[0] in only]
todo = [(i, s) for i, s in enumerate(shots) if only or not done(s[0], s[1], s[2])]
print("%d shots to render (%d already done), %d at a time" % (len(todo), len(shots) - len(todo), JOBS), flush=True)
from concurrent.futures import ThreadPoolExecutor
# the heaviest scenes (big crowds / terrain) go first so they don't end up running together at the end
heavy = ("s15", "s14", "s13", "s09b", "s16c", "s23")
todo.sort(key=lambda it: 0 if it[1][0] in heavy else 1)
with ThreadPoolExecutor(max_workers=JOBS) as ex:
    for msg in ex.map(work, todo):
        print(msg, flush=True)
# contact sheets in edit order (7 shots per sheet)
W, H = 480, 270
rows = []
for sid, a, b, _ in TL.SHOTS:
    ims = []
    for tag in "amz":
        p = os.path.join(FR, "%s_%s.jpg" % (sid, tag))
        im = cv2.imread(p) if os.path.exists(p) else None
        ims.append(cv2.resize(im, (W, H)) if im is not None else np.zeros((H, W, 3), np.uint8))
    lab = np.zeros((H, 150, 3), np.uint8)
    cv2.putText(lab, sid, (8, 50), 0, 1.0, (0, 220, 255), 2)
    cv2.putText(lab, "%.2fs" % ((b - a) / 24), (8, 90), 0, 0.6, (200, 200, 200), 1)
    rows.append(np.hstack([lab] + ims))
for k in range(0, len(rows), 7):
    cv2.imwrite(os.path.join(OUT, "sheet_%d.jpg" % (k // 7 + 1)), np.vstack(rows[k:k + 7]), [cv2.IMWRITE_JPEG_QUALITY, 88])
shutil.copytree(FR, os.path.join(OUT, "frames"), dirs_exist_ok=True)
print("\nDone in %.0f min. Sheets: %s" % ((time.time() - t_all) / 60, OUT))
repo = os.path.dirname(ROOT)
subprocess.run(["git", "add", os.path.join(ROOT, "review")], cwd=repo)
subprocess.run(["git", "commit", "-m", "review: stills of every shot (first/mid/last)"], cwd=repo)
subprocess.run(["git", "push"], cwd=repo)
subprocess.run(["open", OUT])
