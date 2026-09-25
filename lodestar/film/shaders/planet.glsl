// ============================================================================================
// HOME — an earth-like world with real single-scattering atmosphere
// ============================================================================================
uniform vec3  uSunDir;
uniform vec3  uSunCol;
uniform float uSunSize;      // angular radius (radians)
uniform vec3  uPlanetPos;
uniform mat3  uPlanetRot;
uniform float uCloudTime;
uniform float uCity;
uniform vec3  uTrailA;
uniform vec3  uTrailB;
uniform float uTrail;
uniform float uStarBright;
uniform vec3  uMilkyAxis;

const float RP = 1.0;
const float RA = 1.032;
const float HR = 0.0048;
const float HM = 0.0013;
const vec3 BETA_R = vec3(10.5, 24.5, 60.0);
const vec3 BETA_M = vec3(19.0);
const float G_MIE = 0.78;

vec2 lightDepth(vec3 p) {
    // optical depth from p toward the sun (rayleigh, mie). Soft planet shadow folded in.
    vec2 a = sphIntersect(p, uSunDir, uPlanetPos, RA);
    float tl = max(a.y, 0.0);
    float ds = tl / 6.0;
    vec2 od = vec2(0.0);
    for (int j = 0; j < 6; j++) {
        vec3 q = p + uSunDir * (ds * (float(j) + 0.5));
        float h = length(q - uPlanetPos) - RP;
        if (h < -0.002) return vec2(1e4);
        h = max(h, 0.0);
        od += vec2(exp(-h / HR), exp(-h / HM)) * ds;
    }
    return od;
}

vec3 atmosphere(vec3 ro, vec3 rd, float tmax, out vec3 T) {
    T = vec3(1.0);
    vec2 a = sphIntersect(ro, rd, uPlanetPos, RA);
    if (a.y < 0.0) return vec3(0.0);
    float t0 = max(a.x, 0.0);
    float t1 = min(a.y, tmax);
    if (t1 <= t0) return vec3(0.0);
    const int N = 18;
    float ds = (t1 - t0) / float(N);
    vec2 od = vec2(0.0);
    vec3 sumR = vec3(0.0), sumM = vec3(0.0);
    for (int i = 0; i < N; i++) {
        vec3 p = ro + rd * (t0 + ds * (float(i) + 0.5));
        float h = max(length(p - uPlanetPos) - RP, 0.0);
        vec2 d = vec2(exp(-h / HR), exp(-h / HM)) * ds;
        od += d;
        vec2 odl = lightDepth(p);
        vec3 tau = BETA_R * (od.x + odl.x) + BETA_M * 1.1 * (od.y + odl.y);
        vec3 att = exp(-tau);
        sumR += d.x * att;
        sumM += d.y * att;
    }
    T = exp(-(BETA_R * od.x + BETA_M * 1.1 * od.y));
    float mu = dot(rd, uSunDir);
    float pr = 3.0 / (16.0 * PI) * (1.0 + mu * mu);
    float g2 = G_MIE * G_MIE;
    float pm = 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + mu * mu)) / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * G_MIE * mu, 1.5));
    return uSunCol * (sumR * BETA_R * pr + sumM * BETA_M * pm);
}

float cloudDensity(vec3 q) {
    vec3 w = tnoise4(q * 1.3 + vec3(0.0, 0.0, uCloudTime * 0.02)).xyz - 0.5;
    vec3 s = q * 3.2 + w * 1.4 + vec3(uCloudTime * 0.01, 0.0, 0.0);
    float c = fbm(s, 8);
    float lat = q.y;
    // storm tracks and the equatorial band
    float bands = 0.55 + 0.25 * exp(-lat * lat * 40.0) + 0.2 * exp(-pow(abs(lat) - 0.7, 2.0) * 60.0);
    c = (c - 0.5) * 2.4 + 0.5;
    return smoothstep(0.5, 0.7, c * bands + 0.12);
}

vec3 surface(vec3 pw, vec3 rd, float tHit) {
    vec3 n = normalize(pw - uPlanetPos);
    vec3 q = uPlanetRot * n;
    vec3 warp = tnoise4(q * 1.1 + 11.0).xyz - 0.5;
    float cont = fbm(q * 1.6 + warp * 1.1, 9);
    float land = smoothstep(0.515, 0.525, cont);
    float elev = clamp((cont - 0.52) * 7.0, 0.0, 1.0);
    float moist = fbm(q * 3.0 + 5.0, 5);
    float lat = abs(q.y);

    vec3 green = vec3(0.045, 0.085, 0.03);
    vec3 dry = vec3(0.23, 0.17, 0.1);
    vec3 desert = vec3(0.36, 0.28, 0.18);
    vec3 lc = mix(desert, green, smoothstep(0.36, 0.52, moist + 0.25 * (lat - 0.25)));
    lc = mix(lc, dry, smoothstep(0.35, 0.75, elev) * 0.7);
    lc = mix(lc, vec3(0.85), smoothstep(0.62, 0.9, elev + 0.2 * lat));
    float ice = smoothstep(0.78, 0.84, lat + 0.06 * (fbm(q * 8.0, 4) - 0.5));
    vec3 ocean = mix(vec3(0.004, 0.018, 0.045), vec3(0.01, 0.07, 0.09), smoothstep(0.47, 0.515, cont));
    vec3 alb = mix(ocean, lc, land);
    alb = mix(alb, vec3(0.9, 0.93, 0.96), ice);

    float cd = cloudDensity(q);
    // cloud shadows: sample slightly toward the sun
    vec3 qs = uPlanetRot * normalize(n + uSunDir * 0.006);
    float cs = cloudDensity(qs);

    vec2 odl = lightDepth(pw + n * 0.0005);
    vec3 sunT = exp(-(BETA_R * odl.x + BETA_M * 1.1 * odl.y));
    float ndl = dot(n, uSunDir);
    vec3 direct = uSunCol * sunT * max(ndl, 0.0);
    vec3 sky = vec3(0.02, 0.05, 0.12) * uSunCol.b * smoothstep(-0.2, 0.3, ndl) * 0.15;

    vec3 col = alb * (direct * (1.0 - 0.75 * cs) + sky);
    // ocean glint
    // ocean glint: tight lobe, wave-perturbed normal, Fresnel
    vec3 wn = normalize(n + (tnoise4(q * 900.0 + uCloudTime * 0.05).xyz - 0.5) * 0.06);
    float fres = 0.02 + 0.98 * pow(1.0 - max(dot(wn, -rd), 0.0), 5.0);
    float spec = ggx(wn, -rd, uSunDir, 0.07) * (1.0 - land) * (1.0 - ice) * (1.0 - cd) * fres;
    col += uSunCol * sunT * min(spec, 40.0) * max(ndl, 0.0) * 0.25;

    // clouds on top
    float cl = max(dot(n, uSunDir) * 0.8 + 0.2, 0.0);
    vec3 cloudCol = vec3(0.95) * (direct * 0.9 + uSunCol * sunT * cl * 0.1 * step(0.0, ndl + 0.1) + sky * 2.0);
    col = mix(col, cloudCol, cd);

    // city lights on the night side
    float night = smoothstep(0.02, -0.18, ndl);
    float pop = smoothstep(0.45, 0.7, fbm(q * 7.0 + 2.0, 5)) * land * (1.0 - ice);
    float cities = smoothstep(0.56, 0.86, tnoise(q * 160.0)) * (0.35 + 0.65 * smoothstep(0.4, 0.8, tnoise(q * 520.0 + 3.0))) * 3.0;
    col += vec3(1.0, 0.62, 0.28) * pop * cities * night * (1.0 - 0.85 * cd) * uCity * 0.03;
    return col;
}

vec3 background(vec3 rd) {
    vec3 c = starfield(rd, 0.35) * uStarBright;
    c += milkyWay(rd, uMilkyAxis) * uStarBright;
    return c;
}

void main() {
    vec3 ro, rd;
    cameraRay(ro, rd);
    vec2 hp = sphIntersect(ro, rd, uPlanetPos, RP);
    float tPlanet = hp.x > 0.0 ? hp.x : 1e9;
    vec2 ship = wrenTrace(ro, rd, tPlanet);
    float tHit = ship.x > 0.0 ? ship.x : tPlanet;

    vec3 T;
    vec3 scat = atmosphere(ro, rd, tHit, T);
    vec3 col;
    if (ship.x > 0.0) {
        vec3 p = ro + rd * ship.x;
        vec2 sh = sphIntersect(p, uSunDir, uPlanetPos, RP * 1.003);
        vec3 sunC = sh.x > 0.0 ? vec3(0.0) : uSunCol * exp(-(BETA_R * lightDepth(p).x));
        vec3 toPlanet = normalize(uPlanetPos - p);
        vec3 fill = vec3(0.1, 0.2, 0.4) * 0.35 * uSunCol.b * max(dot(-toPlanet, uSunDir) * 0.5 + 0.5, 0.1);
        col = shadeWren(p, rd, ship.y, uSunDir, sunC * 0.9, fill, -toPlanet) * T + scat;
    } else if (hp.x > 0.0) {
        col = surface(ro + rd * hp.x, rd, hp.x) * T + scat;
    } else {
        col = background(rd) * T;
        float ang = acos(clamp(dot(rd, uSunDir), -1.0, 1.0));
        float disc = smoothstep(uSunSize, uSunSize * 0.85, ang);
        // lens glow only when the sun itself is visible
        vec3 toP = uPlanetPos - ro;
        float dp = length(toP);
        float sep = acos(clamp(dot(uSunDir, toP / dp), -1.0, 1.0)) - asin(clamp(RP / dp, 0.0, 1.0));
        float vis = smoothstep(-uSunSize, uSunSize, sep);
        col += uSunCol * T * (disc * 300.0 + vis * (0.8 * exp(-ang * 90.0) + 0.01 / (1.0 + pow(ang / 0.03, 2.0))));
        col += scat;
    }
    col += wrenGlow(ro, rd, ship.x > 0.0 ? ship.x : (hp.x > 0.0 ? hp.x : -1.0));

    if (uTrail > 0.0) {
        vec3 rs = raySegment(ro, rd, uTrailA, uTrailB);
        if (rs.y < tPlanet) {
            float d = length(uTrailB - uTrailA);
            float w = mix(0.0012, 0.006, rs.z) * d;
            float core = exp(-rs.x * rs.x / (w * w * 0.2));
            float halo = exp(-rs.x * rs.x / (w * w * 3.0));
            vec3 c = mix(vec3(1.0, 0.95, 0.85), vec3(1.0, 0.35, 0.08), rs.z);
            col += c * (core * 4.0 + halo * 0.6) * pow(1.0 - rs.z, 1.5) * uTrail;
        }
    }
    fragColor = vec4(col, 1.0);
}
