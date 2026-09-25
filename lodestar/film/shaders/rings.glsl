// ============================================================================================
// THE RINGS — a banded gas giant with a ring system; the Wren skims the ring plane.
// Planet radius 1 at uGiantPos, rings in the planet's local XZ plane.
// ============================================================================================
uniform vec3  uSunDir;
uniform vec3  uSunCol;
uniform vec3  uGiantPos;
uniform mat3  uGiantRot;     // world -> planet local
uniform float uRockOn;
uniform vec2  uRockBase;     // integer cell index of the camera (planet-local xz)
uniform vec2  uRockFrac;     // camera offset inside that cell
uniform float uCamLocalY;    // camera height above the ring plane
#define ROCK_CELL 0.0006
uniform float uStarBright;

const float RIN = 1.32;
const float ROUT = 2.35;

float ringDensity(float r) {
    if (r < RIN || r > ROUT) return 0.0;
    float x = (r - RIN) / (ROUT - RIN);
    float d = 0.55 + 0.45 * sin(r * 37.0) * sin(r * 11.3 + 1.0);
    d *= 0.6 + 0.4 * hash11(floor(r * 900.0));
    d *= 0.75 + 0.25 * hash11(floor(r * 4000.0) + 7.0);
    d *= smoothstep(0.0, 0.05, x) * smoothstep(1.0, 0.9, x);
    // gaps: a Cassini-like division and a few thin ones
    d *= 1.0 - 0.95 * exp(-pow((r - 1.95) * 60.0, 2.0));
    d *= 1.0 - 0.8 * exp(-pow((r - 1.62) * 180.0, 2.0));
    d *= 1.0 - 0.7 * exp(-pow((r - 2.2) * 150.0, 2.0));
    d *= mix(0.8, 1.2, smoothstep(1.45, 1.8, r));
    return clamp(d, 0.0, 1.0);
}

vec3 ringColor(float r) {
    vec3 a = vec3(0.72, 0.62, 0.5), b = vec3(0.36, 0.3, 0.26), c = vec3(0.85, 0.8, 0.72);
    float n = 0.5 + 0.5 * sin(r * 23.0 + 0.7 * sin(r * 5.0));
    float m = hash11(floor(r * 260.0));
    return mix(mix(b, a, n), c, smoothstep(2.0, 2.3, r) * 0.5) * (0.7 + 0.5 * m);
}

vec3 giantSurface(vec3 n, vec3 rd, vec3 wp) {
    vec3 q = uGiantRot * n;
    float lat = q.y;
    vec3 w = tnoise4(q * 3.0 + vec3(uTime * 0.004, 0.0, 0.0)).xyz - 0.5;
    float tl = lat + 0.05 * w.x + 0.03 * (fbm(q * 9.0 + w * 2.0, 5) - 0.5);
    float band = sin(tl * 34.0) * 0.5 + 0.5;
    float band2 = sin(tl * 13.0 + 1.3) * 0.5 + 0.5;
    vec3 cream = vec3(0.9, 0.8, 0.6), tan = vec3(0.68, 0.45, 0.26), rust = vec3(0.5, 0.24, 0.12), pale = vec3(0.55, 0.62, 0.68);
    vec3 col = mix(tan, cream, band);
    col = mix(col, rust, smoothstep(0.55, 0.9, band2) * 0.6);
    col = mix(col, pale, smoothstep(0.75, 0.95, abs(lat)));
    // turbulent swirls along the belts
    float turb = fbm(vec3(q.x * 6.0, tl * 40.0, q.z * 6.0) + w * 3.0, 6);
    col *= 0.85 + 0.3 * turb;
    // a great storm
    vec3 sc = normalize(vec3(0.6, -0.32, 0.74));
    float ds = length(q - sc);
    col = mix(col, vec3(0.75, 0.38, 0.22), smoothstep(0.14, 0.05, ds) * (0.6 + 0.4 * sin(ds * 80.0)));

    float ndl = dot(n, uSunDir);
    float diff = max(ndl, 0.0);
    // ring shadow on the planet
    vec3 lp = uGiantRot * (wp - uGiantPos);
    vec3 ls = uGiantRot * uSunDir;
    if (abs(ls.y) > 1e-4) {
        float tr = -lp.y / ls.y;
        if (tr > 0.0) {
            vec3 rp = lp + ls * tr;
            diff *= 1.0 - 0.85 * ringDensity(length(rp.xz));
        }
    }
    float limb = pow(max(dot(n, -rd), 0.0), 0.35);
    vec3 c = col * col * uSunCol * diff * limb * 0.9;
    // thin haze at the terminator + rim
    float rim = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
    c += vec3(0.5, 0.6, 0.8) * uSunCol * rim * smoothstep(-0.2, 0.4, ndl) * 0.25;
    c += col * 0.004;
    return c;
}

// tumbling ice/rock chunks in a slab around the ring plane, near the camera
float rockField(vec3 p, out float id) {
    // p is camera-relative; the grid is anchored at the camera for float precision
    vec3 lr = uGiantRot * p;
    float cell = ROCK_CELL;
    vec2 q = lr.xz + uRockFrac;
    vec2 gl = floor(q / cell);
    vec2 g = gl + uRockBase;
    vec2 f = q - (gl + 0.5) * cell;
    vec3 lp = vec3(0.0, lr.y + uCamLocalY, 0.0);
    id = hash21(mod(g, 4096.0));
    vec3 h = hash33(vec3(mod(g, 4096.0), 1.0));
    if (h.z > 0.35) return cell * 0.3;
    float rad = cell * (0.06 + 0.16 * h.x * h.x);
    vec3 c = vec3((h.x - 0.5) * cell * 0.6, (h.y - 0.5) * cell * 0.25, (h.z - 0.5) * cell * 0.6);
    vec3 d = vec3(f.x, lp.y, f.y) - c;
    float ang = uTime * (0.3 + h.y) + h.x * 10.0;
    d.xy = rot2(ang) * d.xy;
    d.yz = rot2(ang * 0.7) * d.yz;
    float disp = (fbm(d / rad * 1.4 + h * 10.0, 3) - 0.5) * 0.7 * rad;
    float s = length(d * vec3(1.0, 0.8, 1.2)) - rad + disp;
    return min(s, cell * 0.3);
}

void main() {
    vec3 ro, rd;
    cameraRay(ro, rd);
    vec3 col = starfield(rd, 0.6) * uStarBright;
    float tHit = 1e9;

    vec2 hp = sphIntersect(ro, rd, uGiantPos, 1.0);
    // ring plane
    vec3 lro = uGiantRot * (ro - uGiantPos);
    vec3 lrd = uGiantRot * rd;
    float tr = abs(lrd.y) > 1e-6 ? -lro.y / lrd.y : -1.0;

    if (hp.x > 0.0) {
        vec3 wp = ro + rd * hp.x;
        col = giantSurface(normalize(wp - uGiantPos), rd, wp);
        tHit = hp.x;
    } else {
        // atmosphere glow around the limb
        vec3 oc = uGiantPos - ro;
        float b = dot(oc, rd);
        float d = sqrt(max(dot(oc, oc) - b * b, 0.0));
        if (b > 0.0) {
            float glow = exp(-(d - 1.0) * 40.0) * step(1.0, d);
            vec3 cp = ro + rd * b;
            float lit = smoothstep(-0.3, 0.5, dot(normalize(cp - uGiantPos), uSunDir));
            col += vec3(0.55, 0.65, 0.9) * glow * lit * uSunCol * 0.3;
        }
    }

    // ice chunks close to the camera (ray-marched in the slab)
    vec2 ship = wrenTrace(ro, rd, tHit);
    float tShip = ship.x > 0.0 ? ship.x : 1e9;
    if (uRockOn > 0.5) {
        float slab = 0.00012;
        float t0 = -1.0, t1 = -1.0;
        if (abs(lrd.y) > 1e-6) {
            float ta = (-slab - lro.y) / lrd.y, tb = (slab - lro.y) / lrd.y;
            t0 = max(min(ta, tb), 0.0);
            t1 = min(max(ta, tb), 0.03);
        } else if (abs(lro.y) < slab) { t0 = 0.0; t1 = 0.03; }
        if (t1 > t0) {
            float t = t0;
            float id;
            for (int i = 0; i < 90; i++) {
                vec3 p = ro + rd * t;
                float h = rockField(p, id);
                if (h < 0.0000015 + t * 0.0004) {
                    if (t < min(tHit, tShip)) {
                        // shade the chunk: estimate a normal
                        vec2 e = vec2(0.000004, 0.0);
                        float i2;
                        vec3 n = normalize(vec3(rockField(p + e.xyy, i2) - rockField(p - e.xyy, i2),
                                                rockField(p + e.yxy, i2) - rockField(p - e.yxy, i2),
                                                rockField(p + e.yyx, i2) - rockField(p - e.yyx, i2)));
                        vec3 alb = mix(vec3(0.75, 0.73, 0.7), vec3(0.45, 0.4, 0.36), id);
                        float dif = max(dot(n, uSunDir), 0.0);
                        float spec = pow(max(dot(reflect(rd, n), uSunDir), 0.0), 30.0) * 0.5;
                        vec3 bounce = vec3(0.9, 0.75, 0.55) * 0.05 * max(-n.y * sign(lro.y), 0.0);
                        col = alb * (uSunCol * dif + bounce) + uSunCol * spec * 0.3;
                        tHit = t;
                    }
                    break;
                }
                t += max(h * 0.9, 0.000002);
                if (t > t1) break;
            }
        }
    }

    // ring plane (in front of the planet or anything behind)
    if (tr > 0.0 && tr < tHit && tr < tShip) {
        vec3 rp = lro + lrd * tr;
        float r = length(rp.xz);
        // filter the ring detail by distance so it never aliases
        float fw = tr * pixelAngle() / max(abs(lrd.y), 0.02);
        float dens = 0.0;
        for (int k = -2; k <= 2; k++) dens += ringDensity(r + float(k) * fw * 0.5);
        dens /= 5.0;
        if (dens > 0.0) {
            vec3 wp = ro + rd * tr;
            // planet shadow on the rings
            vec2 sh = sphIntersect(wp, uSunDir, uGiantPos, 1.0);
            float shadow = sh.x > 0.0 ? 0.04 : 1.0;
            vec3 lsun = uGiantRot * uSunDir;
            float sameSide = sign(lsun.y) * sign(lro.y);
            float mu = dot(rd, uSunDir);
            float fwd = 0.35 + 1.4 * pow(max(mu, 0.0), 6.0);
            float lit = sameSide > 0.0 ? 1.0 : (1.0 - dens) * 1.6 * fwd;
            vec3 rc = ringColor(r) * uSunCol * lit * shadow * 0.32;
            float alpha = clamp(dens * dens * 1.4, 0.0, 0.96);
            col = mix(col, rc, alpha);
        }
    }

    if (ship.x > 0.0 && ship.x <= tHit) {
        vec3 p = ro + rd * ship.x;
        vec3 fill = vec3(0.9, 0.78, 0.6) * 0.12 * uSunCol.r;
        vec3 ringN = transpose(uGiantRot) * vec3(0.0, 1.0, 0.0);
        col = shadeWren(p, rd, ship.y, uSunDir, uSunCol, fill, ringN * sign(lro.y));
    }
    col += wrenGlow(ro, rd, ship.x > 0.0 ? ship.x : (tHit < 1e8 ? tHit : -1.0));

    // the sun
    float ang = acos(clamp(dot(rd, uSunDir), -1.0, 1.0));
    if (hp.x < 0.0) col += uSunCol * (smoothstep(0.006, 0.005, ang) * 200.0 + 0.6 * exp(-ang * 80.0));
    fragColor = vec4(col, 1.0);
}
