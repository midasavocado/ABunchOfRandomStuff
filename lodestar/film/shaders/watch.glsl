// ============================================================================================
// THE WATCH — a skeleton pocket-watch movement, macro photography style.
// Units: millimetres. The dial plane is XZ, +Y points up toward the crystal.
// ============================================================================================
uniform float uSec;       // stepped second count (the second hand / escape wheel)
uniform float uBal;       // balance wheel angle (radians)
uniform float uCrack;     // crack growth 0..1
uniform vec3  uLampPos;
uniform vec3  uLampCol;
uniform vec3  uEnvCol;    // environment (softbox) tint and strength
uniform vec3  uRimCol;
uniform float uDust;
uniform float uSpin;      // extra free rotation for the gear train (time running wild)

// ---- materials
#define M_PLATE 1.0
#define M_GOLD 2.0
#define M_RUBY 3.0
#define M_BLUE 4.0
#define M_STEEL 5.0
#define M_DIAL 6.0
#define M_CASE 7.0
#define M_INDEX 8.0

float sdBox2(vec2 p, vec2 b) { vec2 d = abs(p) - b; return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0); }

float extrude(float d2, float y, float h, float r) {
    vec2 w = vec2(d2 + r, abs(y) - h + r);
    return min(max(w.x, w.y), 0.0) + length(max(w, 0.0)) - r;
}

// 2D skeletonised gear. p in gear space.
float gear2D(vec2 p, float R, float teeth, float th, float rimW, float hubR, float spokes, float spokeW) {
    float r = length(p);
    float a = atan(p.y, p.x);
    float seg = TAU / teeth;
    float aa = mod(a + seg * 0.5, seg) - seg * 0.5;
    vec2 q = r * vec2(cos(aa), sin(aa));
    float halfW = R * seg * 0.26;
    float tooth = sdBox2(q - vec2(R + th * 0.35, 0.0), vec2(th * 0.5, halfW * (1.0 - 0.35 * clamp((q.x - R) / th, 0.0, 1.0))));
    float outer = min(r - R, tooth);
    float inner = r - (R - rimW);
    float hub = hubR - r;
    float ss = TAU / spokes;
    float sa = mod(a, ss) - ss * 0.5;
    vec2 sq = r * vec2(cos(sa), sin(sa));
    float spoke = spokeW * 0.5 - abs(sq.y);
    float hole = max(max(inner, hub), spoke);
    hole = max(hole, -(r - hubR - 0.6));
    return max(outer, -hole);
}

float gear(vec3 p, vec2 c, float y0, float h, float ang, float R, float teeth, float spokes) {
    vec3 q = p - vec3(c.x, y0, c.y);
    float bound = length(q.xz) - R - 0.6;
    float by = abs(q.y) - h - 0.1;
    if (max(bound, by) > 0.5) return max(bound, by);
    q.xz = rot2(ang) * q.xz;
    float d2 = gear2D(q.xz, R, teeth, min(0.55, R * 0.12), max(0.45, R * 0.13), max(0.55, R * 0.16), spokes, max(0.35, R * 0.09));
    float d = extrude(d2, q.y, h, 0.03);
    // arbor (axle) + pinion
    float arbor = sdCylY(q, 0.3, h + 0.9);
    return min(d, arbor);
}

float spiral(vec2 p, float r0, float r1, float pitch, float rotIn) {
    float r = length(p);
    float w = clamp((r - r0) / (r1 - r0), 0.0, 1.0);
    float a = atan(p.y, p.x) - rotIn * (1.0 - w);
    float k = (r - r0) / pitch - a / TAU;
    float d = abs(fract(k + 0.5) - 0.5) * pitch;
    d = max(d, r0 - r);
    d = max(d, r - r1);
    return d;
}

// gear train geometry
const vec2 C_CENTER = vec2(0.0, 0.0);
const vec2 C_THIRD = vec2(9.6, 4.8);
const vec2 C_FOURTH = vec2(7.4, -6.2);
const vec2 C_ESC = vec2(2.0, -11.2);
const vec2 C_BAL = vec2(-7.6, -9.6);
const vec2 C_BARREL = vec2(-8.8, 5.2);

vec2 mapWatch(vec3 p) {
    float sec = uSec;
    vec2 res = vec2(1e9, 0.0);
    float r = length(p.xz);

    // case + back
    float caseD = max(abs(r - 20.4) - 1.1, abs(p.y - 1.4) - 3.4);
    caseD = min(caseD, max(r - 20.5, abs(p.y + 2.6) - 0.6));
    caseD -= 0.25;
    if (caseD < res.x) res = vec2(caseD, M_CASE);

    // main plate with cut-outs where the train shows through
    float plate = max(r - 19.6, abs(p.y + 1.1) - 0.7);
    float cut = min(min(length(p.xz - C_BAL) - 6.4, length(p.xz - C_ESC) - 3.2), length(p.xz - C_THIRD) - 4.9);
    cut = min(cut, length(p.xz - C_FOURTH) - 4.4);
    plate = max(plate, -cut);
    if (plate < res.x) res = vec2(plate, M_PLATE);
    float lower = max(r - 19.6, abs(p.y + 3.0) - 0.4);
    if (lower < res.x) res = vec2(lower, M_PLATE);

    // gear train
    float spin = uSpin;
    float gc = gear(p, C_CENTER, 0.25, 0.18, sec * TAU / 900.0 + spin * 0.1, 7.2, 80.0, 5.0);
    float g3 = gear(p, C_THIRD, 0.75, 0.15, -sec * TAU / 240.0 - spin * 0.25, 4.3, 50.0, 4.0);
    float g4 = gear(p, C_FOURTH, 0.95, 0.14, sec * TAU / 60.0 + spin * 0.6, 3.9, 48.0, 4.0);
    float ge = gear(p, C_ESC, 1.2, 0.12, -sec * TAU / 30.0 - spin * 1.2, 2.5, 15.0, 3.0);
    float gb = gear(p, C_BARREL, 0.35, 0.3, sec * TAU / 7200.0 + spin * 0.05, 6.4, 96.0, 6.0);
    float g = min(min(gc, g3), min(min(g4, ge), gb));
    if (g < res.x) res = vec2(g, M_GOLD);

    // balance wheel: rim, arms, timing screws, hairspring
    vec3 b = p - vec3(C_BAL.x, 1.7, C_BAL.y);
    if (length(b.xz) < 6.5 && abs(b.y) < 1.6) {
        vec3 bq = b;
        bq.xz = rot2(uBal) * bq.xz;
        float rb = length(bq.xz);
        float rim = extrude(abs(rb - 5.0) - 0.32, bq.y, 0.28, 0.06);
        float aa = mod(atan(bq.z, bq.x), TAU / 3.0) - TAU / 6.0;
        vec2 aq = rb * vec2(cos(aa), sin(aa));
        float arms = extrude(max(abs(aq.y) - 0.28, rb - 4.9), bq.y, 0.16, 0.05);
        float sa = mod(atan(bq.z, bq.x), TAU / 12.0) - TAU / 24.0;
        vec2 sq = rb * vec2(cos(sa), sin(sa));
        float screws = length(vec3(sq.x - 5.45, bq.y, sq.y)) - 0.26;
        float bal = min(min(rim, arms), sdCylY(bq, 0.28, 1.4));
        if (bal < res.x) res = vec2(bal, M_GOLD);
        if (screws < res.x) res = vec2(screws, M_GOLD);
        vec3 hq = b - vec3(0.0, -0.55, 0.0);
        float hs = max(spiral(hq.xz, 0.7, 3.9, 0.36, uBal * 1.2) - 0.06, abs(hq.y) - 0.12);
        if (hs < res.x) res = vec2(hs, M_BLUE);
    }

    // bridges (with Geneva stripes, shaded later) and the balance cock
    vec3 bp = p - vec3(0.0, 2.05, 0.0);
    // train bridge: a curved arm over third, fourth and escape wheels
    vec2 e1 = C_THIRD, e2 = C_FOURTH, e3 = C_ESC;
    float br = sdCapsule(vec3(p.x, 0.0, p.z), vec3(e1.x, 0.0, e1.y), vec3(e2.x, 0.0, e2.y), 1.35);
    br = min(br, sdCapsule(vec3(p.x, 0.0, p.z), vec3(e2.x, 0.0, e2.y), vec3(e3.x, 0.0, e3.y), 1.2));
    br = min(br, sdCapsule(vec3(p.x, 0.0, p.z), vec3(e1.x, 0.0, e1.y), vec3(16.5, 0.0, 9.0), 1.6));
    br = min(br, sdCapsule(vec3(p.x, 0.0, p.z), vec3(e3.x, 0.0, e3.y), vec3(6.0, 0.0, -17.5), 1.5));
    float bridge = extrude(br, bp.y, 0.32, 0.12);
    // balance cock
    float ck = sdCapsule(vec3(p.x, 0.0, p.z), vec3(C_BAL.x, 0.0, C_BAL.y), vec3(-15.5, 0.0, -11.5), 1.25);
    ck = min(ck, length(p.xz - C_BAL) - 1.7);
    float cock = extrude(ck, p.y - 2.75, 0.3, 0.12);
    // barrel bridge
    float bb = min(length(p.xz - C_BARREL) - 3.0, sdCapsule(vec3(p.x, 0.0, p.z), vec3(C_BARREL.x, 0.0, C_BARREL.y), vec3(-15.0, 0.0, 11.0), 2.2));
    float bbr = extrude(bb, p.y - 1.25, 0.3, 0.12);
    float bridges = min(min(bridge, cock), bbr);
    if (bridges < res.x) res = vec2(bridges, M_PLATE);

    // jewels on the bridges
    float jw = 1e9;
    jw = min(jw, sdCylY(p - vec3(C_THIRD.x, 2.42, C_THIRD.y), 0.55, 0.1));
    jw = min(jw, sdCylY(p - vec3(C_FOURTH.x, 2.42, C_FOURTH.y), 0.55, 0.1));
    jw = min(jw, sdCylY(p - vec3(C_ESC.x, 2.42, C_ESC.y), 0.5, 0.1));
    jw = min(jw, sdCylY(p - vec3(C_BAL.x, 3.12, C_BAL.y), 0.6, 0.1));
    jw = min(jw, sdCylY(p - vec3(C_BARREL.x, 1.62, C_BARREL.y), 0.7, 0.1));
    jw -= 0.05;
    if (jw < res.x) res = vec2(jw, M_RUBY);

    // blued screws
    float sc = 1e9;
    vec2 S[6] = vec2[6](vec2(16.5, 9.0), vec2(6.0, -17.5), vec2(-15.5, -11.5), vec2(-15.0, 11.0), vec2(4.4, 0.2), vec2(-4.2, 10.4));
    float SY[6] = float[6](2.4, 2.4, 3.1, 1.6, 2.4, 1.6);
    for (int i = 0; i < 6; i++) {
        vec3 sp = p - vec3(S[i].x, SY[i], S[i].y);
        if (length(sp) > 2.0) continue;
        float head = max(length(sp * vec3(1.0, 1.6, 1.0)) / 1.3 - 0.62, -sp.y);
        vec2 sl = rot2(float(i) * 1.3) * sp.xz;
        head = max(head, -(max(abs(sl.x) - 0.1, sp.y - 0.25)));
        sc = min(sc, head);
    }
    if (sc < res.x) res = vec2(sc, M_BLUE);

    // chapter ring with applied indices
    float ring = extrude(abs(r - 17.8) - 1.6, p.y - 3.0, 0.1, 0.05);
    if (ring < res.x) res = vec2(ring, M_DIAL);
    float ia = mod(atan(p.z, p.x) + TAU / 24.0, TAU / 12.0) - TAU / 24.0;
    vec2 iq = r * vec2(cos(ia), sin(ia));
    float idx = extrude(sdBox2(iq - vec2(17.9, 0.0), vec2(1.1, 0.22)), p.y - 3.18, 0.08, 0.04);
    if (idx < res.x) res = vec2(idx, M_INDEX);

    // hands: hour, minute, seconds (blued steel), central cannon
    vec3 hp = p;
    float hub = sdCylY(hp - vec3(0.0, 3.6, 0.0), 0.75, 0.55);
    // hour hand (~10:08)
    vec2 hq = rot2(-2.2) * hp.xz;
    float hh = extrude(max(sdBox2(hq - vec2(4.4, 0.0), vec2(4.4, 0.55 - 0.25 * max(hq.x - 5.0, 0.0) / 4.0)), -0.2 - hq.x), hp.y - 3.55, 0.05, 0.03);
    // minute hand
    vec2 mq = rot2(-4.35 - uSec * TAU / 3600.0) * hp.xz;
    float mh = extrude(max(sdBox2(mq - vec2(7.0, 0.0), vec2(7.0, 0.38 - 0.28 * max(mq.x, 0.0) / 14.0)), -0.2 - mq.x), hp.y - 3.75, 0.045, 0.02);
    // seconds hand with counterweight
    vec2 sq2 = rot2(PI * 0.5 - uSec * TAU / 60.0) * hp.xz;
    float sh = extrude(sdBox2(sq2 - vec2(6.2, 0.0), vec2(10.4, 0.1)), hp.y - 3.95, 0.035, 0.015);
    sh = min(sh, extrude(length(sq2 + vec2(3.2, 0.0)) - 0.7, hp.y - 3.95, 0.04, 0.02));
    float hands = min(min(hh, mh), min(sh, hub));
    if (hands < res.x) res = vec2(hands, M_BLUE);
    return res;
}

vec3 watchNormal(vec3 p) {
    const vec2 k = vec2(1.0, -1.0);
    const float e = 0.004;
    return normalize(k.xyy * mapWatch(p + k.xyy * e).x + k.yyx * mapWatch(p + k.yyx * e).x +
                     k.yxy * mapWatch(p + k.yxy * e).x + k.xxx * mapWatch(p + k.xxx * e).x);
}

vec3 envStudio(vec3 r) {
    // a dark room: big warm softbox above-left, cool strip on the right, faint floor bounce
    // overhead softbox (soft-edged rectangle), a far low softbox for the flat plates,
    // a cool strip light on the right and a warm dome
    vec3 d1 = normalize(vec3(-0.35, 0.9, 0.2));
    float sb = smoothstep(0.62, 0.8, dot(r, d1));
    vec3 d2 = normalize(vec3(0.15, 0.38, 1.0));
    vec2 q2 = vec2(dot(r, normalize(cross(d2, vec3(0.0, 1.0, 0.0)))), r.y - d2.y);
    float far = smoothstep(0.5, 0.35, abs(q2.x)) * smoothstep(0.2, 0.08, abs(q2.y)) * step(0.0, dot(r, d2));
    float strip = smoothstep(0.9, 0.96, dot(r, normalize(vec3(0.9, 0.3, -0.3))));
    float dome = 0.06 + 0.12 * smoothstep(-0.1, 0.8, r.y);
    return uEnvCol * (sb * 2.5 + far * 1.6 + dome) + uRimCol * strip * 1.8;
}

float softShadow(vec3 ro, vec3 rd, float tmax) {
    float res = 1.0, t = 0.03;
    for (int i = 0; i < 40; i++) {
        float h = mapWatch(ro + rd * t).x;
        res = min(res, 8.0 * h / t);
        t += clamp(h, 0.02, 0.8);
        if (res < 0.002 || t > tmax) break;
    }
    return clamp(res, 0.0, 1.0);
}

float calcAO(vec3 p, vec3 n) {
    float o = 0.0, s = 1.0;
    for (int i = 1; i <= 5; i++) {
        float h = 0.06 + 0.22 * float(i);
        o += (h - mapWatch(p + n * h).x) * s;
        s *= 0.75;
    }
    return clamp(1.0 - 0.9 * o, 0.0, 1.0);
}

vec3 shade(vec3 p, vec3 rd, float mat) {
    vec3 n = watchNormal(p);
    vec3 v = -rd;
    vec3 alb = vec3(0.7);
    float rough = 0.3, metal = 1.0;
    float r = length(p.xz);
    if (mat == M_PLATE) {
        alb = vec3(0.72, 0.73, 0.76);
        // perlage on the plate, Geneva stripes on the bridges
        if (p.y < 0.0) {
            vec2 cell = floor(p.xz / 1.1);
            vec2 f = fract(p.xz / 1.1) - 0.5;
            float ring = fract(length(f) * 9.0 + hash21(cell));
            rough = 0.22 + 0.18 * ring;
            alb *= 0.9 + 0.1 * ring;
        } else {
            float stripe = fract((p.x * 0.7 + p.z * 0.7) / 2.2);
            float s = smoothstep(0.0, 0.5, stripe) * smoothstep(1.0, 0.5, stripe);
            rough = 0.12 + 0.22 * s;
            alb *= 0.92 + 0.12 * s;
        }
    } else if (mat == M_GOLD) {
        alb = vec3(1.0, 0.76, 0.4);
        float circ = fract(r * 6.0);
        rough = 0.16 + 0.12 * circ;
    } else if (mat == M_BLUE) {
        alb = vec3(0.1, 0.22, 0.75);
        rough = 0.12;
    } else if (mat == M_DIAL) {
        alb = vec3(0.86, 0.85, 0.82);
        rough = 0.32;
        // minute track
        float a = atan(p.z, p.x) / TAU * 60.0;
        float tick = smoothstep(0.08, 0.02, abs(fract(a) - 0.5) - 0.43) * step(abs(r - 16.7), 0.35);
        alb = mix(alb, vec3(0.02), tick);
        metal = 0.6;
    } else if (mat == M_INDEX) {
        alb = vec3(1.0, 0.8, 0.45); rough = 0.1;
    } else if (mat == M_CASE) {
        alb = vec3(1.0, 0.8, 0.5); rough = 0.08;
    }

    vec3 L = uLampPos - p;
    float ld = length(L);
    L /= ld;
    float att = 1.0 / (1.0 + ld * ld * 0.0006);
    float sh = softShadow(p + n * 0.02, L, ld);
    float ao = calcAO(p, n);
    float ndl = max(dot(n, L), 0.0);
    vec3 F0 = mix(vec3(0.04), alb, metal);
    vec3 h = normalize(L + v);
    float fres = pow(1.0 - max(dot(n, v), 0.0), 5.0);
    vec3 F = F0 + (1.0 - F0) * fres;
    vec3 spec = uLampCol * att * sh * ndl * ggx(n, v, L, rough) * F * 3.0;
    vec3 diff = (1.0 - metal) * alb * uLampCol * att * sh * ndl / PI * 1.2;
    vec3 refl = reflect(rd, n);
    vec3 envc = envStudio(refl) * F * (1.0 - rough * 0.6) * ao;
    envc *= 0.3 + 0.7 * softShadow(p + n * 0.02, refl, 6.0);
    vec3 col = diff + spec + envc + (1.0 - metal) * alb * uEnvCol * 0.02 * ao;

    if (mat == M_RUBY) {
        // synthetic ruby: glassy surface + light glowing inside
        vec3 ruby = vec3(0.9, 0.03, 0.08);
        float inner = pow(max(dot(-rd, L) * 0.5 + 0.5, 0.0), 2.0);
        col = ruby * (0.04 + uLampCol * att * sh * 0.1 * (0.3 + inner)) + envStudio(refl) * (0.05 + 0.95 * fres) * 0.7;
        col += uLampCol * att * sh * ggx(n, v, L, 0.05) * 0.6;
    }
    return col;
}

// crystal crack pattern on the plane y = 4.6
float crackLines(vec2 q) {
    vec2 c0 = vec2(6.5, -4.5);
    vec2 d = q - c0;
    float r = length(d);
    float a = atan(d.y, d.x);
    float lines = 0.0;
    for (int i = 0; i < 9; i++) {
        float fi = float(i);
        float ai = fi * TAU / 9.0 + 0.35 * sin(fi * 7.1);
        float len = (5.0 + 14.0 * hash11(fi * 3.7)) * uCrack;
        float wob = 0.35 * sin(r * 1.7 + fi * 3.0) * min(r, 3.0) / 3.0;
        vec2 dir = vec2(cos(ai), sin(ai));
        float along = dot(d, dir);
        float off = abs(dot(d, vec2(-dir.y, dir.x)) + wob);
        float m = step(0.0, along) * step(along, len);
        lines = max(lines, m * smoothstep(0.045, 0.0, off) * (1.0 - 0.6 * along / max(len, 0.01)));
    }
    // concentric fracture arcs near the impact
    float ringD = abs(fract(r / 2.3 + 0.5) - 0.5) * 2.3;
    float arcs = smoothstep(0.045, 0.0, ringD) * step(1.0, r) * step(r, 7.5 * uCrack) * step(0.55, fract(a * 1.3 + floor(r / 2.3 + 0.5) * 0.37));
    return max(lines, arcs * 0.7) * step(0.001, uCrack);
}

vec3 dustMotes(vec3 ro, vec3 rd, float tHit) {
    vec3 col = vec3(0.0);
    if (uDust <= 0.0) return col;
    for (int i = 0; i < 48; i++) {
        float fi = float(i);
        vec3 h = hash33(vec3(fi, fi * 1.7, 3.1));
        vec3 pos = uCamPos + uCamRot * vec3((h.x - 0.5) * 14.0, (h.y - 0.5) * 7.0, 2.0 + h.z * 18.0);
        pos += vec3(sin(uTime * 0.13 + fi) * 0.8, uTime * 0.05 * (0.5 + h.y), cos(uTime * 0.11 + fi * 2.0) * 0.8);
        vec3 rp = raySegment(ro, rd, pos, pos + 1e-4);
        if (rp.y > tHit) continue;
        float coc = abs(rp.y - uFocus) * 0.05 + 0.04;
        float a = exp(-rp.x * rp.x / (coc * coc));
        float lit = max(dot(normalize(uLampPos - pos), -rd) * 0.5 + 0.5, 0.0);
        col += uLampCol * a * lit * 0.012 / (coc * 6.0 + 0.1) * (0.4 + h.x);
    }
    return col * uDust;
}

void main() {
    vec3 ro, rd;
    cameraRay(ro, rd);
    float tmax = 90.0;
    float t = 0.0;
    vec2 h = vec2(0.0);
    bool hit = false;
    // start the march at the crystal/case bounding volume
    vec2 bs = sphIntersect(ro, rd, vec3(0.0, 0.0, 0.0), 24.0);
    if (bs.y > 0.0) {
        t = max(bs.x, 0.0);
        tmax = bs.y;
        for (int i = 0; i < 200; i++) {
            h = mapWatch(ro + rd * t);
            if (h.x < 0.0015 * t + 0.002) { hit = true; break; }
            t += h.x * 0.85;
            if (t > tmax) break;
        }
    }
    vec3 col;
    if (hit) {
        col = shade(ro + rd * t, rd, h.y);
    } else {
        col = envStudio(rd) * 0.03;
    }
    float tHit = hit ? t : 1e9;

    // the crystal: faint reflection + crack
    if (rd.y < 0.0 && ro.y > 4.6) {
        float tc = (4.6 - ro.y) / rd.y;
        vec3 pc = ro + rd * tc;
        if (length(pc.xz) < 19.5 && tc < tHit) {
            float fres = 0.04 + 0.96 * pow(1.0 - abs(rd.y), 5.0);
            vec3 refl = reflect(rd, vec3(0.0, 1.0, 0.0));
            col = col * (1.0 - fres) + envStudio(refl) * fres * 0.8;
            float c = crackLines(pc.xz);
            if (c > 0.0) {
                float glint = pow(max(dot(normalize(uLampPos - pc), refl), 0.0), 2.0);
                col = col * (1.0 - 0.3 * c) + (uLampCol * 0.25 * (0.4 + glint) + envStudio(refl) * 1.2 + 0.04) * c;
            }
        }
    }
    col += dustMotes(ro, rd, tHit);
    fragColor = vec4(col, 1.0);
}
