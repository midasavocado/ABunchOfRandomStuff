"""Debug camera override for lookdev: SB_DBGCAM="x,y,z,tx,ty,tz,lens[,fstop]" replaces the shot camera."""
import os, bpy
import sb


def apply():
    s = os.environ.get("SB_DBGCAM")
    if not s:
        return
    v = [float(t) for t in s.split(",")]
    cam = sb.camera("DbgCam", loc=v[0:3], target=v[3:6], lens=v[6], fstop=(v[7] if len(v) > 7 else None))
    cam.data.clip_start = max(1e-5, (cam.location - __import__('mathutils').Vector(v[3:6])).length / 200)
    cam.data.clip_end = 400000.0
    bpy.context.scene.camera = cam
