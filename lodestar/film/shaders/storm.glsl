// ============================================================================================
// THE STORM — a volumetric nebula storm: lightning, a vortex, and later the lodestar's light.
// The camera flies roughly along +z. One unit ~ a few ship lengths.
// ============================================================================================
uniform vec3  uColA;        // hot emission colour
uniform vec3  uColB;        // secondary emission colour
uniform vec3  uDustCol;     // scattering albedo of the dust
uniform float uDensity;
uniform float uEmission;
uniform float uWallZ;       // the storm begins at this world z (soft, carved edge)
uniform float uClearZ;      // the storm ends at this world z (for the breakthrough)
uniform float uVortex;      // 0..1
uniform vec3  uVortexPos;
uniform vec3  uVortexAxis;  // points from the eye toward the viewer
uniform float uSpin;
uniform vec3  uStarDir;
uniform vec3  uStarCol;
uniform float uStarI;
uniform vec3  uFlashPos[3];
uniform vec3  uFlashCol;
uniform float uFlashI[3];
uniform vec3  uBolt[8];     // one jagged bolt polyline
uniform float uBoltI;
uniform float uStarBright;
uniform float uSteps;
uniform vec4  uBlob;       // finite storm mass: centre + radius (w = 0 -> infinite)
uniform vec3  uEyeCol;
uniform float uNoiseScale;

float stormDensity(vec3 p, out float heat) {
    heat = 0.0;
    vec3 q = p * 0.33 * uNoiseScale;
    float swirl = 0.0;
    float vmask = 1.0;
    float eye = 0.0;
    if (uVortex > 0.0) {
        vec3 v = p - uVortexPos;
        float h = dot(v, uVortexAxis);
        vec3 radial = v - uVortexAxis * h;
        float r = length(radial);
        float Rf = 1.6 + 0.2 * max(h, 0.0);
        // spin the lookup around the axis: faster near the eye
        vec3 ax = uVortexAxis;
        vec3 b1 = normalize(cross(ax, abs(ax.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
        vec3 b2 = cross(ax, b1);
        float ang = atan(dot(radial, b2), dot(radial, b1) + 1e-5);
        float tw = uSpin * (1.0 + 1.5 / (r + 1.0)) + 0.9 * log(r + 0.5) + h * 0.04;
        float a2 = ang + tw;
        // sample the noise in a frame that co-rotates with the flow -> long spiral arms
        vec3 sw = vec3(cos(a2) * r * 0.38, sin(a2) * r * 0.38, h * 0.1);
        q = mix(q, sw + vec3(3.0), uVortex);
        float wall = exp(-pow((r - Rf) / (0.9 + 0.12 * max(h, 0.0)), 2.0));
        vmask = mix(1.0, 0.25 + 1.3 * wall, uVortex) * mix(1.0, smoothstep(0.35 * Rf, 0.9 * Rf, r), uVortex);
        eye = uVortex * exp(-r * r / 0.6) * smoothstep(8.0, -6.0, h);
    }
    vec3 w = tnoise4(q * 0.5 + vec3(0.0, uTime * 0.015, 0.0)).xyz - 0.5;
    float d = fbm(q + w * 1.8, 5);
    d = smoothstep(0.47, 0.68, d);
    float detail = fbm(q * 3.1 + w * 2.6 + 4.0, 4);
    d *= smoothstep(0.25, 0.75, detail) * 1.6;
    heat = fbm(q * 0.7 + 9.0 + w, 3);
    d *= vmask;
    if (uBlob.w > 0.0) {
        float en = (fbm(p * 0.08 + 2.0, 5) - 0.5) * 30.0;
        d *= smoothstep(uBlob.w + 3.0, uBlob.w - 8.0, length(p - uBlob.xyz) + en);
    }
    // the storm's outer wall and far edge
    float edgeN = (fbm(vec3(p.xy * 0.18, 3.0), 4) - 0.5) * 7.0;
    d *= smoothstep(uWallZ - 1.5, uWallZ + 2.5, p.z + edgeN);
    d *= smoothstep(uClearZ + 1.5, uClearZ - 3.0, p.z - edgeN * 0.8);
    heat += eye * 3.0;
    return d * uDensity;
}

void main() {
    vec3 ro, rd;
    cameraRay(ro, rd);
    vec2 ship = wrenTrace(ro, rd, 1e9);
    float tShip = ship.x > 0.0 ? ship.x : 1e9;

    // background: stars (visible outside the storm), and the lodestar
    vec3 bg = starfield(rd, 0.8) * uStarBright;
    float sa = acos(clamp(dot(rd, uStarDir), -1.0, 1.0));
    bg += uStarCol * uStarI * (smoothstep(0.012, 0.009, sa) * 60.0 + 0.9 * exp(-sa * 60.0) + 0.08 * exp(-sa * 9.0));

    // precompute bolt segments against this ray
    vec3 boltRS[7];
    for (int i = 0; i < 7; i++) boltRS[i] = uBoltI > 0.0 ? raySegment(ro, rd, uBolt[i], uBolt[i + 1]) : vec3(1e9);

    vec3 col = vec3(0.0);
    float T = 1.0;
    float t = 0.05;
    int N = int(uSteps);
    float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    ign = fract(ign + fract(uTime * 24.0) * 0.61803);
    float tmax = 60.0;
    float mu = dot(rd, uStarDir);
    float g = 0.65;
    float hgStar = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * mu, 1.5) * 0.25;
    for (int i = 0; i < 160; i++) {
        if (i >= N || T < 0.01 || t > tmax) break;
        float ds = 0.05 + t * 0.032;
        float tt = t + ds * ign;
        if (tt > tShip) {
            // the ship is inside the volume: stop here, the ship shades below
            break;
        }
        vec3 p = ro + rd * tt;
        float heat;
        float d = stormDensity(p, heat);
        // lightning bolts glow through the cloud
        for (int b = 0; b < 7; b++) {
            if (boltRS[b].y >= t && boltRS[b].y < t + ds) {
                float r2 = boltRS[b].x * boltRS[b].x;
                col += T * uFlashCol * uBoltI * (0.03 / (r2 * 900.0 + 0.03) + 0.2 * exp(-r2 * 30.0));
            }
        }
        if (d > 0.002) {
            float sigma = d * 1.6;
            // emission (ionised gas) + scattered light
            vec3 emis = mix(uColB, uColA, smoothstep(0.35, 0.7, heat)) * (uEmission * (0.08 + 2.6 * pow(smoothstep(0.38, 0.8, heat), 2.0)));
            vec3 li = vec3(0.0);
            for (int f = 0; f < 3; f++) {
                if (uFlashI[f] <= 0.0) continue;
                vec3 lv = uFlashPos[f] - p;
                float l2 = dot(lv, lv);
                li += uFlashCol * uFlashI[f] / (1.0 + l2 * 0.35);
            }
            if (uStarI > 0.0) {
                float h2;
                float occ = stormDensity(p + uStarDir * 0.9, h2) + stormDensity(p + uStarDir * 2.4, h2);
                li += uStarCol * uStarI * exp(-occ * 2.5) * (0.03 + 0.6 * hgStar);
            }
            vec3 S = (emis + uDustCol * li) * sigma;
            float Tr = exp(-sigma * ds);
            col += T * S * (1.0 - Tr) / max(sigma, 1e-4);
            T *= Tr;
        }
        t += ds;
    }
    if (ship.x > 0.0 && T > 0.01) {
        vec3 p = ro + rd * ship.x;
        // light on the ship: nearest flash, the star, and the storm glow
        vec3 key = vec3(0.0);
        vec3 kd = uStarDir;
        float kmax = 0.0;
        for (int f = 0; f < 3; f++) {
            vec3 lv = uFlashPos[f] - p;
            float I = uFlashI[f] / (1.0 + dot(lv, lv) * 0.2);
            if (I > kmax) { kmax = I; kd = normalize(lv); }
        }
        vec3 sunC = uStarCol * uStarI * 0.6;
        vec3 fill = mix(uColB, uColA, 0.5) * uEmission * 0.35;
        vec3 sc;
        if (kmax > uStarI * 0.6) sc = shadeWren(p, rd, ship.y, kd, uFlashCol * kmax * 0.6, fill, -rd);
        else sc = shadeWren(p, rd, ship.y, uStarDir, sunC, fill, -rd);
        col += T * sc;
        T = 0.0;
    }
    if (uVortex > 0.0) {
        vec3 toE = normalize(uVortexPos - ro);
        float ea = acos(clamp(dot(rd, toE), -1.0, 1.0));
        bg += uEyeCol * (exp(-ea * 40.0) * 6.0 + exp(-ea * 9.0) * 0.8) * uVortex;
    }
    col += T * bg;
    col += wrenGlow(ro, rd, ship.x);
    fragColor = vec4(col, 1.0);
}
