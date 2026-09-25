// ============================================================================================
// THE VOID — lost in the dark: a dead ship tumbling, a heartbeat, then a single star.
// Also the backdrop for the title card.
// ============================================================================================
uniform vec3  uStarDir;
uniform vec3  uStarCol;
uniform float uStarI;
uniform float uStarBright;
uniform vec3  uRimDir;
uniform vec3  uRimCol;
uniform float uDebris;

void main() {
    vec3 ro, rd;
    cameraRay(ro, rd);
    vec3 col = starfield(rd, 0.35) * uStarBright;
    col += milkyWay(rd, normalize(vec3(0.2, 1.0, 0.4))) * uStarBright * 0.6;
    // the lodestar
    float sa = acos(clamp(dot(rd, uStarDir), -1.0, 1.0));
    col += uStarCol * uStarI * (smoothstep(0.004, 0.0025, sa) * 80.0 + exp(-sa * 120.0) * 1.5 + exp(-sa * 18.0) * 0.12);

    // drifting debris / ice glints from the storm
    if (uDebris > 0.0) {
        for (int i = 0; i < 40; i++) {
            float fi = float(i);
            vec3 h = hash33(vec3(fi, 7.0, 3.0));
            vec3 pos = uShipPos + (h - 0.5) * vec3(6.0, 4.0, 6.0) * uShipScale * 3.0;
            pos += (hash33(vec3(fi, 1.0, 9.0)) - 0.5) * uTime * 0.02 * uShipScale;
            vec3 rp = raySegment(ro, rd, pos, pos + 1e-5);
            float glint = pow(max(sin(uTime * (1.0 + h.x * 3.0) + fi), 0.0), 8.0);
            float r2 = rp.x * rp.x / (uShipScale * uShipScale);
            col += uRimCol * (0.2 + glint) * uDebris * 0.0004 / (r2 + 0.0004);
        }
    }

    vec2 ship = wrenTrace(ro, rd, 1e9);
    if (ship.x > 0.0) {
        vec3 p = ro + rd * ship.x;
        col = shadeWren(p, rd, ship.y, uRimDir, uRimCol, vec3(0.004, 0.006, 0.01), vec3(0.0, 1.0, 0.0));
        // the lodestar lights the ship as it grows
        col += shadeWren(p, rd, ship.y, uStarDir, uStarCol * uStarI * 0.4, vec3(0.0), vec3(0.0, 1.0, 0.0)) * step(0.001, uStarI);
    }
    col += wrenGlow(ro, rd, ship.x);
    fragColor = vec4(col, 1.0);
}
