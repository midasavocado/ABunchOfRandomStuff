# People — MANDATORY approach (client feedback: primitive/SDF mannequin people looked terrible)

Never show primitive, metaball, skin-modifier or SDF mannequin humans on screen. Use the MPFB (MakeHuman for
Blender) generator that is now installed (extension `bl_ext.user_default.mpfb`, CC0 MakeHuman assets installed),
via `lib/mhchild.py`:

```python
import mhchild
bm, rig, parts = mhchild.build(name="Mom", phenotype=dict(age=0.52, gender=0.0, weight=0.5,
                               race=dict(african=0.2, asian=0.5, caucasian=0.3)),
                               hair="long01", clothes=["female_casualsuit01", "shoes02"], skin="young_asian_female")
mhchild.fix_eyes(parts)            # optional: amber-brown irises (the film's child uses this)
mhchild.pose_bone(rig, "upperarm01.L", rot=(x, y, z))   # degrees; rig = MPFB 'default' skeleton
```
- Age slider: 0.0=1 y, 0.1875=11 y, 0.5=25 y, 1.0=90 y. Height ~1.22 m at age 0.155.
- Hair: afro01, bob01, bob02, braid01, long01, ponytail01, short01..short04.
- Clothes: male_casualsuit01..06, female_casualsuit01/02, female_elegantsuit01, female_sportsuit01, male_elegantsuit01,
  male_worksuit01, fedora01, shoes01..06. AVOID garments with printed logos (casualsuit02/04 tees, female_casualsuit01/02 tees)
  or recolor/cover them — the film must have no logos. You may recolor via material node edits (see recolor_top()).
- Skins: young/middleage/old × african/asian/caucasian × male/female (skins/<name>/<name>.mhmat).
- Bones (default rig): root, pelvis.L/R, upperleg01/02, lowerleg01/02, foot, spine01..05, neck01..03, head, clavicle,
  shoulder01, upperarm01/02, lowerarm01/02, wrist, finger1-1..finger5-3 (.L/.R). Pose with rotations; keyframe
  pose bones for animation (walks: animate upperleg/lowerleg/foot + root translation with foot planting — no sliding).
- The film's child: `mhchild.build()` defaults + `fix_eyes` + `recolor_top` + `add_cuffs` (amber cuffs). Only the
  director uses the child.
- Cycles or EEVEE both fine. Faces must hold up: light them well, or keep them soft/partially framed if a shot demands.
- Build time is ~20-40 s per person; cache with `sb.save()`/linked duplicates for crowds of 3-6 (vary phenotype/clothes).
