"""Sequential final-render queue. Append lines 'sid' to queue.txt; renders missing frames at 4K."""
import os, sys, time, subprocess
sys.path.insert(0, "lib")
import timeline as TL
ROOT = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(ROOT, "queue.txt")
LOG = os.path.join(ROOT, "queue.log")
def done(sid):
    if os.path.exists(os.path.join(ROOT, "edit", sid + ".mov")):
        return True
    _, a, b, _ = TL.shot(sid)
    d = os.path.join(ROOT, "renders", sid)
    return all(os.path.exists(os.path.join(d, "%04d.png" % f)) for f in range(a, b))
while True:
    todo = [l.strip() for l in open(Q) if l.strip() and not l.startswith("#")] if os.path.exists(Q) else []
    todo = [s for s in todo if not done(s)]
    if not todo:
        if os.path.exists(os.path.join(ROOT, "queue.stop")): break
        time.sleep(20); continue
    sid = todo[0]
    scene = TL.shot(sid)[3]
    t = time.time()
    with open(LOG, "a") as lg:
        lg.write(f"{time.strftime('%H:%M:%S')} START {sid}\n"); lg.flush()
        env = dict(os.environ, SB_RES="final")
        r = subprocess.run(["blender", "-b", "-P", f"scenes/{scene}.py", "--", sid], cwd=ROOT, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        lg.write(f"{time.strftime('%H:%M:%S')} END {sid} rc={r.returncode} {time.time()-t:.0f}s {r.stderr[-400:] if r.returncode else ''}\n")
    if done(sid):
        pr = subprocess.run(["python3", "post.py", sid], cwd=ROOT, capture_output=True, text=True)
        with open(LOG, "a") as lg:
            lg.write(f"{time.strftime('%H:%M:%S')} POST {sid} {pr.stdout.strip()[-200:]} {pr.stderr.strip()[-300:]}\n")
    if r.returncode != 0 and not done(sid):
        # avoid hot loop on a broken scene: comment it out
        lines = open(Q).read().replace(sid + "\n", "#FAILED " + sid + "\n")
        open(Q, "w").write(lines)
