// ============================================================================================
// LODESTAR — shared shader library (included at the top of every scene shader)
// ============================================================================================
out vec4 fragColor;

uniform vec2  uRes;        // render target size (pixels)
uniform vec2  uJitter;     // sub-pixel jitter for this sub-frame (pixels)
uniform float uTime;       // global film time (s)
uniform float uLocal;      // time within the shot (s)
uniform vec3  uCamPos;
uniform mat3  uCamRot;     // columns: right, up, forward
uniform float uFov;        // tan(vertical fov / 2)
uniform vec2  uLens;       // lens sample for depth of field (unit disk, per sub-frame)
uniform float uAperture;
uniform float uFocus;
uniform sampler3D uNoise;  // tileable smooth noise, 4 independent channels

#define PI 3.14159265359
#define TAU 6.28318530718

// ---------------------------------------------------------------- hashing / noise
float hash11(float p) { p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
float hash21(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float hash31(vec3 p3) { p3 = fract(p3 * 0.1031); p3 += dot(p3, p3.zyx + 31.32); return fract((p3.x + p3.y) * p3.z); }
vec3 hash33(vec3 p3) {
    p3 = fract(p3 * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}
vec2 hash22(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973)); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.xx + p3.yz) * p3.zy); }

// texture noise: one unit of p ~ one feature; returns roughly [0,1], mean 0.5
float tnoise(vec3 p) { return texture(uNoise, p * 0.125).x; }
vec4 tnoise4(vec3 p) { return texture(uNoise, p * 0.125); }

const mat3 OCT = mat3(0.00, 0.80, 0.60, -0.80, 0.36, -0.48, -0.60, -0.48, 0.64);

float fbm(vec3 p, int oct) {
    float a = 0.5, s = 0.0, n = 0.0;
    for (int i = 0; i < 10; i++) {
        if (i >= oct) break;
        s += a * tnoise(p);
        n += a;
        p = OCT * p * 2.03;
        a *= 0.5;
    }
    return s / n;
}

// ridged fbm (for lightning-carved clouds, cracks, etc.)
float rfbm(vec3 p, int oct) {
    float a = 0.5, s = 0.0, n = 0.0;
    for (int i = 0; i < 8; i++) {
        if (i >= oct) break;
        float v = 1.0 - abs(tnoise(p) * 2.0 - 1.0);
        s += a * v * v;
        n += a;
        p = OCT * p * 2.03;
        a *= 0.5;
    }
    return s / n;
}

// ---------------------------------------------------------------- camera
void cameraRay(out vec3 ro, out vec3 rd) {
    vec2 p = (gl_FragCoord.xy + uJitter - 0.5 * uRes) / (0.5 * uRes.y);
    vec3 dir = normalize(vec3(p * uFov, 1.0));
    vec3 o = vec3(0.0);
    if (uAperture > 0.0) {
        vec3 fp = dir * (uFocus / dir.z);
        o = vec3(uLens * uAperture, 0.0);
        dir = normalize(fp - o);
    }
    ro = uCamPos + uCamRot * o;
    rd = uCamRot * dir;
}

float pixelAngle() { return 2.0 * uFov / uRes.y; }

// ---------------------------------------------------------------- geometry helpers
vec2 sphIntersect(vec3 ro, vec3 rd, vec3 c, float r) {
    vec3 oc = ro - c;
    float b = dot(oc, rd);
    float h = b * b - dot(oc, oc) + r * r;
    if (h < 0.0) return vec2(-1.0);
    h = sqrt(h);
    return vec2(-b - h, -b + h);
}

mat2 rot2(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }

float sdBox(vec3 p, vec3 b) { vec3 q = abs(p) - b; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0); }
float sdRoundBox(vec3 p, vec3 b, float r) { vec3 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r; }
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) { vec3 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h) - r; }
float sdCylZ(vec3 p, float r, float h) { vec2 d = abs(vec2(length(p.xy), p.z)) - vec2(r, h); return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)); }
float sdCylY(vec3 p, float r, float h) { vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h); return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)); }
float sdEllipsoid(vec3 p, vec3 r) { float k0 = length(p / r); float k1 = length(p / (r * r)); return k0 * (k0 - 1.0) / k1; }
float sdRoundCone(vec3 p, vec3 a, vec3 b, float r1, float r2) {
    vec3 ba = b - a; float l2 = dot(ba, ba); float rr = r1 - r2; float a2 = l2 - rr * rr; float il2 = 1.0 / l2;
    vec3 pa = p - a; float y = dot(pa, ba); float z = y - l2; vec3 xv = pa * l2 - ba * y; float x2 = dot(xv, xv);
    float y2 = y * y * l2; float z2 = z * z * l2; float k = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k) return sqrt(x2 + z2) * il2 - r2;
    if (sign(y) * a2 * y2 < k) return sqrt(x2 + y2) * il2 - r1;
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}
float smin(float a, float b, float k) { float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0); return mix(b, a, h) - k * h * (1.0 - h); }

// distance between a ray and a segment; returns (distance, t along ray, s along segment 0..1)
vec3 raySegment(vec3 ro, vec3 rd, vec3 a, vec3 b) {
    vec3 ba = b - a;
    vec3 oa = ro - a;
    float baba = dot(ba, ba), rdba = dot(rd, ba), oard = dot(oa, rd), oaba = dot(oa, ba);
    float den = baba - rdba * rdba;
    float s = den > 1e-6 ? clamp((oaba - oard * rdba) / den, 0.0, 1.0) : 0.0;
    float t = max(dot(a + ba * s - ro, rd), 0.0);
    return vec3(length(ro + rd * t - (a + ba * s)), t, s);
}

// ---------------------------------------------------------------- color
vec3 blackbody(float k) { // approx. colour of a black body at temperature k (Kelvin), normalised
    float t = k / 100.0;
    vec3 c;
    c.r = t <= 66.0 ? 1.0 : clamp(1.292936 * pow(t - 60.0, -0.1332047592), 0.0, 1.0);
    c.g = t <= 66.0 ? clamp(0.3900815 * log(t) - 0.6318414, 0.0, 1.0) : clamp(1.129890 * pow(t - 60.0, -0.0755148492), 0.0, 1.0);
    c.b = t >= 66.0 ? 1.0 : (t <= 19.0 ? 0.0 : clamp(0.5432068 * log(t - 10.0) - 1.19625408, 0.0, 1.0));
    return c;
}

// ---------------------------------------------------------------- star field
vec3 starLayer(vec3 d, float cells, float bright, float seed) {
    vec3 ad = abs(d);
    vec3 fd; vec2 uv; float face;
    if (ad.x >= ad.y && ad.x >= ad.z) { uv = d.yz / ad.x; face = d.x > 0.0 ? 0.0 : 1.0; }
    else if (ad.y >= ad.z) { uv = d.xz / ad.y; face = d.y > 0.0 ? 2.0 : 3.0; }
    else { uv = d.xy / ad.z; face = d.z > 0.0 ? 4.0 : 5.0; }
    vec2 g = (uv * 0.5 + 0.5) * cells;
    vec2 cell = floor(g);
    vec3 h = hash33(vec3(cell, face * 17.0 + seed));
    vec2 sp = cell + 0.2 + 0.6 * h.xy;
    vec2 suv = sp / cells * 2.0 - 1.0;
    vec3 sd;
    if (face < 0.5) sd = vec3(1.0, suv.x, suv.y);
    else if (face < 1.5) sd = vec3(-1.0, suv.x, suv.y);
    else if (face < 2.5) sd = vec3(suv.x, 1.0, suv.y);
    else if (face < 3.5) sd = vec3(suv.x, -1.0, suv.y);
    else if (face < 4.5) sd = vec3(suv.x, suv.y, 1.0);
    else sd = vec3(suv.x, suv.y, -1.0);
    sd = normalize(sd);
    float ang = acos(clamp(dot(d, sd), -1.0, 1.0));
    float sig = pixelAngle() * 0.65;
    float b = pow(h.z, 16.0) * bright;
    vec3 col = blackbody(mix(3200.0, 11000.0, hash31(vec3(cell, face + seed + 3.0))));
    return col * b * exp(-0.5 * ang * ang / (sig * sig));
}

vec3 milkyWay(vec3 d, vec3 axis) {
    float band = dot(d, axis);
    float w = exp(-band * band * 18.0);
    vec3 q = d * 3.0;
    float n = fbm(q + 7.0, 6);
    float dust = smoothstep(0.42, 0.62, fbm(q * 2.0 + 3.0, 6));
    vec3 col = mix(vec3(0.55, 0.62, 0.85), vec3(1.0, 0.82, 0.62), smoothstep(0.3, 0.7, n));
    return col * w * (0.25 + 1.2 * n * n) * (1.0 - 0.8 * dust * w) * 0.035;
}

vec3 starfield(vec3 d, float density) {
    vec3 c = vec3(0.0);
    c += starLayer(d, 18.0, 40.0, 1.0);
    c += starLayer(d, 45.0, 14.0, 2.0);
    c += starLayer(d, 110.0, 5.0, 3.0) * density;
    c += starLayer(d, 240.0, 2.0, 4.0) * density;
    return c;
}

// ============================================================================================
// THE WREN — our ship. Local space: nose +z, up +y. Length ~4 units.
// ============================================================================================
uniform vec3  uShipPos;
uniform mat3  uShipRot;     // local -> world
uniform float uShipScale;
uniform float uShipOn;
uniform float uEngine;      // engine glow (0 = dead, 1 = cruise, >1 = burn)
uniform float uShipLights;  // nav lights / cockpit strip
uniform float uHeat;        // re-entry heating 0..1

vec2 wrenMap(vec3 p) {
    // returns (distance, material). 1 hull, 2 dark metal, 3 canopy, 4 nozzle
    // lifting body: wider than tall, flat engine block at the back
    vec3 q = p;
    q.y *= 1.75;
    q.x *= 0.92;
    float hull = sdRoundCone(q, vec3(0.0, 0.0, -1.35), vec3(0.0, 0.0, 2.0), 0.5, 0.035) / 1.75;
    hull = max(hull, -(p.z + 1.62));
    hull = max(hull, -p.y - 0.2);
    // dorsal spine
    float spine = sdCapsule(p, vec3(0.0, 0.16, -1.45), vec3(0.0, 0.1, 0.5), 0.1);
    hull = smin(hull, spine, 0.12);
    // wings: swept, tapered, slight dihedral, blended into the body
    vec3 w = p;
    w.x = abs(w.x);
    w.z += 0.75 * (w.x - 0.3);
    w.y -= 0.08 * (w.x - 0.3) + 0.03;
    float chord = mix(0.62, 0.14, clamp((w.x - 0.3) / 1.5, 0.0, 1.0));
    float wing = sdRoundBox(w - vec3(1.0, -0.06, -0.62), vec3(0.78, 0.016, chord), 0.014);
    // small canted wingtip fins
    vec3 wt = w - vec3(1.72, 0.06, -0.8);
    wt.xy = rot2(-0.35) * wt.xy;
    wt.z += 0.5 * wt.y;
    float tip = sdRoundBox(wt, vec3(0.012, 0.14, 0.16), 0.01);
    // engine nacelles, fused into the body
    vec3 e = p;
    e.x = abs(e.x);
    float nac = sdCapsule(e, vec3(0.5, -0.05, -1.5), vec3(0.5, -0.05, -0.1), 0.15);
    nac = max(nac, -(e.z + 1.58));
    float noz = sdCylZ(e - vec3(0.5, -0.05, -1.6), 0.125, 0.05);
    noz = max(noz, -sdCylZ(e - vec3(0.5, -0.05, -1.66), 0.1, 0.05));
    // dorsal fin
    vec3 f = p;
    f.z += 0.9 * (f.y - 0.25);
    float fin = sdRoundBox(f - vec3(0.0, 0.46, -1.15), vec3(0.014, 0.22, mix(0.36, 0.12, clamp((p.y - 0.25) / 0.45, 0.0, 1.0))), 0.01);
    // canopy, set into the spine
    float can = sdEllipsoid(p - vec3(0.0, 0.13, 0.62), vec3(0.14, 0.085, 0.42));

    float body = smin(hull, wing, 0.1);
    body = smin(body, nac, 0.12);
    vec2 r = vec2(body, 1.0);
    if (nac < hull && nac < wing + 0.01) r.y = 2.0;
    if (tip < r.x) r = vec2(tip, 1.0);
    if (fin < r.x) r = vec2(fin, 1.0);
    if (noz < r.x) r = vec2(noz, 4.0);
    if (can < r.x) r = vec2(can, 3.0);
    return r;
}

vec2 wrenWorld(vec3 pw) {
    vec3 p = transpose(uShipRot) * (pw - uShipPos) / uShipScale;
    vec2 r = wrenMap(p);
    r.x *= uShipScale;
    return r;
}

vec3 wrenNormal(vec3 p) {
    const vec2 k = vec2(1.0, -1.0);
    float e = 0.0015 * uShipScale;
    return normalize(k.xyy * wrenWorld(p + k.xyy * e).x + k.yyx * wrenWorld(p + k.yyx * e).x +
                     k.yxy * wrenWorld(p + k.yxy * e).x + k.xxx * wrenWorld(p + k.xxx * e).x);
}

// march the ship; returns (t, material) or (-1)
vec2 wrenTrace(vec3 ro, vec3 rd, float tmax) {
    if (uShipOn < 0.5) return vec2(-1.0);
    vec2 bs = sphIntersect(ro, rd, uShipPos, 2.3 * uShipScale);
    if (bs.y < 0.0) return vec2(-1.0);
    float t = max(bs.x, 0.0);
    float tend = min(bs.y, tmax);
    for (int i = 0; i < 110; i++) {
        vec2 h = wrenWorld(ro + rd * t);
        if (h.x < 0.0004 * t + 0.0005 * uShipScale) return vec2(t, h.y);
        t += h.x * 0.8;
        if (t > tend) break;
    }
    return vec2(-1.0);
}

// panel lines + stripe in ship-local coordinates
vec3 wrenAlbedo(vec3 pw, float mat, out float rough, out float metal) {
    vec3 p = transpose(uShipRot) * (pw - uShipPos) / uShipScale;
    rough = 0.38; metal = 0.0;
    vec3 col = vec3(0.8, 0.81, 0.83);
    if (mat > 1.5 && mat < 2.5) { col = vec3(0.1, 0.105, 0.11); rough = 0.3; metal = 0.5; }
    if (mat > 3.5) { col = vec3(0.04); rough = 0.25; metal = 1.0; }
    if (mat < 2.5) {
        // panel seams
        float sz = abs(fract(p.z * 2.4) - 0.5);
        float sx = abs(fract(p.x * 2.8 + 0.5) - 0.5);
        float seam = 1.0 - 0.45 * (1.0 - smoothstep(0.0, 0.012, 0.5 - sz)) - 0.35 * (1.0 - smoothstep(0.0, 0.012, 0.5 - sx)) * step(0.3, abs(p.x));
        col *= seam;
        col *= 0.93 + 0.07 * hash21(floor(vec2(p.z * 2.4, p.x * 2.8)));
        // belly: dark heat tiles
        float belly = smoothstep(0.02, -0.04, p.y + 0.02 * abs(p.x));
        vec2 tg = abs(fract(vec2(p.x, p.z) * 9.0) - 0.5);
        float tiles = 0.75 + 0.25 * smoothstep(0.44, 0.5, max(tg.x, tg.y));
        vec3 tileCol = vec3(0.045, 0.045, 0.05) * (0.8 + 0.4 * hash21(floor(vec2(p.x, p.z) * 9.0))) / tiles;
        col = mix(col, tileCol, belly * step(mat, 1.5));
        rough = mix(rough, 0.7, belly);
        // amber flank stripe, dark nose tip, registration band
        float stripe = smoothstep(0.02, 0.0, abs(p.y - 0.035 + 0.03 * p.z) - 0.018) * step(abs(p.x), 0.45) * step(-1.3, p.z) * step(p.z, 1.4);
        col = mix(col, vec3(0.95, 0.45, 0.08), stripe * step(mat, 1.5));
        col = mix(col, vec3(0.06), smoothstep(1.6, 1.72, p.z));
        col = mix(col, vec3(0.9, 0.42, 0.06), smoothstep(0.015, 0.0, abs(p.z + 1.25) - 0.03) * step(0.3, p.y + 0.15 * step(0.02, abs(p.x))));
        col *= 0.92 + 0.08 * fbm(p * 5.0, 3);
    }
    return col;
}

// GGX-ish specular
float ggx(vec3 n, vec3 v, vec3 l, float r) {
    vec3 h = normalize(v + l);
    float a = r * r;
    float nh = max(dot(n, h), 0.0);
    float d = a * a / (PI * pow(nh * nh * (a * a - 1.0) + 1.0, 2.0));
    return d * 0.25;
}

// shades the ship. env() is supplied by each scene through the macro WREN_ENV
vec3 shadeWren(vec3 p, vec3 rd, float mat, vec3 sunDir, vec3 sunCol, vec3 fillCol, vec3 fillDir) {
    vec3 n = wrenNormal(p);
    float rough, metal;
    vec3 alb = wrenAlbedo(p, mat, rough, metal);
    vec3 v = -rd;
    // cheap AO
    float ao = 0.0, sca = 1.0;
    for (int i = 1; i <= 4; i++) {
        float h = 0.03 * float(i) * uShipScale;
        ao += (h - wrenWorld(p + n * h).x) * sca;
        sca *= 0.7;
    }
    ao = clamp(1.0 - 3.0 * ao / uShipScale, 0.2, 1.0);
    // self shadow toward the sun
    float sh = 1.0;
    float t = 0.02 * uShipScale;
    for (int i = 0; i < 28; i++) {
        float h = wrenWorld(p + sunDir * t).x;
        sh = min(sh, 10.0 * h / t);
        t += clamp(h, 0.01 * uShipScale, 0.3 * uShipScale);
        if (sh < 0.01 || t > 4.0 * uShipScale) break;
    }
    sh = clamp(sh, 0.0, 1.0);
    float ndl = max(dot(n, sunDir), 0.0);
    vec3 spec = sunCol * ggx(n, v, sunDir, rough) * ndl * sh * mix(vec3(0.04), alb, metal) * 4.0;
    vec3 diff = alb * (1.0 - metal) * (sunCol * ndl * sh + fillCol * (0.5 + 0.5 * dot(n, fillDir)) * ao + vec3(0.004) * ao);
    // fresnel rim
    float fr = pow(1.0 - max(dot(n, v), 0.0), 5.0);
    vec3 col = diff + spec + fillCol * fr * 0.5 * ao;
    // backlight rim when looking toward the sun
    col += sunCol * pow(1.0 - max(dot(n, v), 0.0), 4.0) * pow(max(dot(rd, sunDir), 0.0), 3.0) * 0.35 * (0.3 + 0.7 * sh);
    if (mat > 2.5 && mat < 3.5) {
        // canopy: dark glass, reflections, faint warm interior light strip
        col = vec3(0.01) + sunCol * ggx(n, v, sunDir, 0.08) * ndl * 0.3 + fillCol * fr;
        col += vec3(1.0, 0.6, 0.25) * 0.25 * uShipLights * smoothstep(0.05, 0.0, abs(dot(n, vec3(0.0, 1.0, 0.0)) - 0.55));
    }
    if (mat > 3.5) {
        vec3 lp = transpose(uShipRot) * (p - uShipPos) / uShipScale;
        float r = length(vec2(abs(lp.x) - 0.5, lp.y + 0.05));
        col += vec3(0.35, 0.65, 1.0) * uEngine * 6.0 * smoothstep(0.1, 0.02, r) * step(lp.z, -1.58);
    }
    // re-entry heating: hull glows from the leading surfaces
    if (uHeat > 0.0) {
        vec3 fwd = uShipRot * vec3(0.0, 0.0, 1.0);
        float lead = max(dot(n, fwd), 0.0);
        col += blackbody(1600.0 + 1400.0 * uHeat) * uHeat * uHeat * 3.0 * pow(lead, 1.5);
    }
    return col;
}

// engine plumes + navigation lights, added along the view ray (occluded beyond tHit)
vec3 wrenGlow(vec3 ro, vec3 rd, float tHit) {
    if (uShipOn < 0.5) return vec3(0.0);
    vec3 col = vec3(0.0);
    float s = uShipScale;
    for (int k = 0; k < 2; k++) {
        float sx = k == 0 ? 0.5 : -0.5;
        vec3 a = uShipPos + uShipRot * (vec3(sx, -0.05, -1.64) * s);
        float len = (0.7 + 1.6 * uEngine) * s;
        vec3 b = a - uShipRot * vec3(0.0, 0.0, 1.0) * len;
        vec3 rs = raySegment(ro, rd, a, b);
        if (rs.y < tHit || tHit < 0.0) {
            float w = s * mix(0.1, 0.03, rs.z);
            float core = exp(-rs.x * rs.x / (w * w * 0.25)) * (1.0 - rs.z);
            float halo = exp(-rs.x * rs.x / (w * w * 4.0)) * pow(1.0 - rs.z, 2.0);
            col += (vec3(0.8, 0.92, 1.0) * core * 3.0 + vec3(0.2, 0.45, 1.0) * halo * 0.8) * uEngine;
            // nozzle flare point
            vec3 rp = raySegment(ro, rd, a, a + 1e-4);
            col += vec3(0.5, 0.75, 1.0) * uEngine * 0.25 * s * s / (rp.x * rp.x + 0.002 * s * s) * 0.02;
        }
    }
    if (uShipLights > 0.0) {
        float blink = step(0.85, fract(uTime * 0.9));
        vec3 L[3] = vec3[3](vec3(1.76, 0.02, -1.05), vec3(-1.76, 0.02, -1.05), vec3(0.0, 0.7, -1.5));
        vec3 C[3] = vec3[3](vec3(0.1, 1.0, 0.3), vec3(1.0, 0.1, 0.05), vec3(1.0) * blink * 3.0);
        for (int i = 0; i < 3; i++) {
            vec3 lp = uShipPos + uShipRot * (L[i] * s);
            vec3 rp = raySegment(ro, rd, lp, lp + 1e-4);
            if (rp.y < tHit + 0.05 * s || tHit < 0.0) {
                float r2 = rp.x * rp.x / (s * s);
                col += C[i] * uShipLights * (0.02 / (r2 * 400.0 + 0.02)) * 0.6;
            }
        }
    }
    return col;
}
