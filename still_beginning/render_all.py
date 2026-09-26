"""Render the whole film and build the master. Run on the Mac (Blender 5.2, Metal).

    python3 render_all.py --preview          # fast full-film check: every shot at 960x540 -> STILL_BEGINNING.mp4 (scaled)
    python3 render_all.py                    # final: every shot at 3840x2160, graded, then the verified master
    python3 render_all.py --shots s09a,s17   # only these shots (final unless --preview)
    python3 render_all.py --master-only      # just re-assemble + verify from existing edit/*.mov

Resumable: a shot whose edit/<sid>.mov (or edit/<sid>_prev.mov) already exists is skipped; delete it to re-render.
Per shot: blender -b -P scenes/<scene>.py -- <sid>  ->  renders/<sid>/NNNN.png (16-bit)  ->  post.py (grade, bloom,
halation, vignette, the s28 title) -> edit/<sid>.mov (H.264 4:4:4 10-bit near-lossless, PNGs deleted after verify).
Log: render_all.log.  Requires: blender on PATH (or BLENDER env var), ffmpeg/ffprobe, python3 with numpy + opencv.
"""
import os, sys, time, subprocess, shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import timeline as TL

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "render_all.log")
BLENDER = os.environ.get("BLENDER") or shutil.which("blender") or "/Applications/Blender.app/Contents/MacOS/Blender"


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def render_shot(sid, scene, preview):
    env = dict(os.environ, SB_RES="preview" if preview else "final", SB_SKIP_EXISTING="1")
    out = os.path.join(ROOT, "edit", sid + ("_prev" if preview else "") + ".mov")
    if os.path.exists(out):
        log(f"skip {sid} ({os.path.basename(out)} exists)")
        return True
    # never mix frames from an older version of a scene: each shot renders into an empty folder
    fdir = os.path.join(ROOT, "preview" if preview else "renders", sid)
    if os.path.isdir(fdir):
        for fn in os.listdir(fdir):
            if fn.endswith(".png"):
                os.remove(os.path.join(fdir, fn))
    t = time.time()
    log(f"render {sid} ({scene})")
    r = subprocess.run([BLENDER, "-b", "-P", os.path.join("scenes", scene + ".py"), "--", sid], cwd=ROOT, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        log(f"FAILED {sid} rc={r.returncode}\n{r.stderr[-2000:]}")
        return False
    log(f"rendered {sid} in {time.time() - t:.0f}s; grading")
    args = [sys.executable, "post.py", sid] + (["--preview"] if preview else [])
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    log(f"post {sid}: {p.stdout.strip()[-200:]} {p.stderr.strip()[-400:]}")
    return p.returncode == 0 and os.path.exists(out)


def main():
    preview = "--preview" in sys.argv
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--shots"):
            only = sys.argv[sys.argv.index(a) + 1].split(",") if a == "--shots" else a.split("=", 1)[1].split(",")
    TL.check()
    if "--master-only" not in sys.argv:
        failed = []
        for sid, a, b, scene in TL.SHOTS:
            if only and sid not in only:
                continue
            if not render_shot(sid, scene, preview):
                failed.append(sid)
        if failed:
            log("FAILED SHOTS: " + ",".join(failed) + " (fix, then re-run: finished shots are skipped)")
            sys.exit(1)
        if only:
            log("done (subset)")
            return
    # the score is generated (deterministic); rebuild it if any audio source is newer than the delivered WAV
    adir = os.path.join(ROOT, "audio")
    wav = os.path.join(adir, "score_master.wav")
    srcs = [os.path.join(adir, f) for f in ("compose.py", "instruments.py", "score.py", "master.py", "sbdsp.py", "dsp.c")]
    if not os.path.exists(wav) or max(os.path.getmtime(f) for f in srcs if os.path.exists(f)) > os.path.getmtime(wav) \
            or "--rebuild-score" in sys.argv:
        log("rebuilding the score (audio/compose.py)")
        r = subprocess.run([sys.executable, "compose.py"], cwd=adir)
        if r.returncode != 0:
            log("score build FAILED"); sys.exit(1)
    log("assembling master")
    r = subprocess.run([sys.executable, "make_master.py"] + (["--preview"] if preview else []), cwd=ROOT)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
