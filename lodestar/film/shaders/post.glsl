// ============================================================================================
// LODESTAR — post-processing. Compiled once per pass with one of the PASS_* defines.
// ============================================================================================
out vec4 fragColor;
uniform sampler2D uSrc;
uniform vec2 uSrcTexel;
uniform vec2 uOutRes;

#ifdef PASS_ACCUM
uniform sampler2D uPrev;
uniform float uW;
uniform float uFirst;
void main() {
    ivec2 ip = ivec2(gl_FragCoord.xy);
    vec4 c = texelFetch(uSrc, ip, 0);
    c = clamp(c, 0.0, 6.0e4);
    vec4 p = uFirst > 0.5 ? vec4(0.0) : texelFetch(uPrev, ip, 0);
    fragColor = p + c * uW;
}
#endif

#ifdef PASS_DOWN
// 13-tap downsample (Jimenez 2014) with optional soft-knee threshold
uniform float uThreshold;
vec3 tap(vec2 uv) { return texture(uSrc, uv).rgb; }
void main() {
    vec2 uv = gl_FragCoord.xy / uOutRes;
    vec2 t = uSrcTexel;
    vec3 a = tap(uv + t * vec2(-2, 2)), b = tap(uv + t * vec2(0, 2)), c = tap(uv + t * vec2(2, 2));
    vec3 d = tap(uv + t * vec2(-2, 0)), e = tap(uv), f = tap(uv + t * vec2(2, 0));
    vec3 g = tap(uv + t * vec2(-2, -2)), h = tap(uv + t * vec2(0, -2)), i = tap(uv + t * vec2(2, -2));
    vec3 j = tap(uv + t * vec2(-1, 1)), k = tap(uv + t * vec2(1, 1)), l = tap(uv + t * vec2(-1, -1)), m = tap(uv + t * vec2(1, -1));
    vec3 col = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
    if (uThreshold > 0.0) {
        float br = max(col.r, max(col.g, col.b));
        float knee = uThreshold * 0.6;
        float soft = clamp(br - uThreshold + knee, 0.0, 2.0 * knee);
        soft = soft * soft / (4.0 * knee + 1e-5);
        float w = max(soft, br - uThreshold) / max(br, 1e-5);
        col *= w;
    }
    fragColor = vec4(col, 1.0);
}
#endif

#ifdef PASS_BLUR
uniform vec2 uDir;
void main() {
    vec2 uv = gl_FragCoord.xy / uOutRes;
    const float W[5] = float[5](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    vec3 c = texture(uSrc, uv).rgb * W[0];
    for (int i = 1; i < 5; i++) {
        c += texture(uSrc, uv + uDir * float(i) * uSrcTexel).rgb * W[i];
        c += texture(uSrc, uv - uDir * float(i) * uSrcTexel).rgb * W[i];
    }
    fragColor = vec4(c, 1.0);
}
#endif

#ifdef PASS_STREAK
// very wide horizontal smear for anamorphic flares
uniform float uSpread;
void main() {
    vec2 uv = gl_FragCoord.xy / uOutRes;
    vec3 c = vec3(0.0);
    float wsum = 0.0;
    for (int i = -24; i <= 24; i++) {
        float w = exp(-abs(float(i)) / 9.0);
        c += texture(uSrc, uv + vec2(float(i) * uSpread * uSrcTexel.x, 0.0)).rgb * w;
        wsum += w;
    }
    fragColor = vec4(c / wsum, 1.0);
}
#endif

#ifdef PASS_RAYS
uniform vec2 uLight;     // light position in uv
uniform float uDecay;
void main() {
    vec2 uv = gl_FragCoord.xy / uOutRes;
    vec2 d = (uv - uLight) / 64.0 * 0.9;
    vec3 c = vec3(0.0);
    float w = 1.0;
    vec2 p = uv;
    for (int i = 0; i < 64; i++) {
        p -= d;
        c += texture(uSrc, p).rgb * w;
        w *= uDecay;
    }
    fragColor = vec4(c / 64.0, 1.0);
}
#endif

#ifdef PASS_FINAL
uniform sampler2D uB1;
uniform sampler2D uB2;
uniform sampler2D uB3;
uniform sampler2D uB4;
uniform sampler2D uStreakTex;
uniform sampler2D uRaysTex;
uniform sampler2D uText;
uniform vec2  uFrameRes;     // full output frame (with letterbox)
uniform float uActiveH;      // active picture height in pixels
uniform float uExposure;
uniform float uBloom;
uniform float uStreak;
uniform vec3  uStreakTint;
uniform float uRays;
uniform vec2  uLight;
uniform float uFlare;
uniform vec3  uFlareTint;
uniform vec3  uTint;
uniform float uSat;
uniform float uContrast;
uniform vec3  uShadowTint;
uniform vec3  uHighTint;
uniform float uVignette;
uniform float uCA;
uniform float uGrain;
uniform float uFade;
uniform float uFlash;
uniform vec3  uFlashCol;
uniform float uTextAlpha;
uniform float uFrame;

vec3 RRTAndODTFit(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}
vec3 aces(vec3 c) {
    const mat3 ACESIn = mat3(0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383, 0.04823, 0.01566, 0.83777);
    const mat3 ACESOut = mat3(1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276, -0.07367, -0.00605, 1.07602);
    c = ACESIn * c;
    c = RRTAndODTFit(c);
    return clamp(ACESOut * c, 0.0, 1.0);
}
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float h12(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }

vec3 ghost(vec2 uv, vec2 pos, float r, vec3 tint) {
    vec2 d = (uv - pos) * vec2(uOutRes.x / uOutRes.y, 1.0);
    float l = length(d);
    return tint * smoothstep(r, r * 0.7, l) * (0.6 + 0.4 * smoothstep(r * 0.4, r, l));
}

void main() {
    // image coordinates: y measured from the top of the frame
    vec2 frag = gl_FragCoord.xy;
    float bar = floor((uFrameRes.y - uActiveH) * 0.5);
    float yTop = frag.y;
    if (yTop < bar || yTop >= bar + uActiveH) { fragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
    vec2 uv = vec2(frag.x / uFrameRes.x, 1.0 - (yTop - bar) / uActiveH);   // scene uv (GL, y up)
    vec2 tuv = vec2(frag.x / uFrameRes.x, yTop / uFrameRes.y);            // text uv (y down)

    vec2 cen = uv - 0.5;
    vec2 caOff = cen * uCA * 0.004;
    vec3 col;
    col.r = texture(uSrc, uv - caOff).r;
    col.g = texture(uSrc, uv).g;
    col.b = texture(uSrc, uv + caOff).b;

    vec3 bloom = texture(uB1, uv).rgb * 0.3 + texture(uB2, uv).rgb * 0.3 + texture(uB3, uv).rgb * 0.24 + texture(uB4, uv).rgb * 0.16;
    col += bloom * uBloom;
    col += texture(uStreakTex, uv).rgb * uStreak * uStreakTint;
    col += texture(uRaysTex, uv).rgb * uRays;

    if (uFlare > 0.0) {
        vec2 L = uLight;
        vec2 axis = vec2(0.5) - L;
        vec3 g = vec3(0.0);
        g += ghost(uv, L + axis * 0.6, 0.035, vec3(0.3, 0.6, 1.0));
        g += ghost(uv, L + axis * 1.2, 0.07, vec3(0.2, 1.0, 0.5) * 0.5);
        g += ghost(uv, L + axis * 1.55, 0.02, vec3(1.0, 0.5, 0.2));
        g += ghost(uv, L + axis * 2.1, 0.07, vec3(0.5, 0.3, 1.0) * 0.3);
        g += ghost(uv, L + axis * 0.3, 0.012, vec3(1.0, 0.8, 0.5));
        // halo ring around the frame centre
        vec2 d = (uv - 0.5) * vec2(uOutRes.x / uOutRes.y, 1.0);
        float ring = exp(-pow((length(d) - 0.42) * 22.0, 2.0));
        g += ring * vec3(0.4, 0.6, 1.0) * 0.03 * smoothstep(0.7, 0.0, length(L - 0.5));
        col += g * uFlare * uFlareTint * 0.35;
    }

    col *= exp2(uExposure);
    col *= uTint;
    float lu = luma(col);
    col = max(mix(vec3(lu), col, uSat), 0.0);
    col = aces(col);

    // grade: contrast around mid-grey + split toning
    col = clamp((col - 0.18) * uContrast + 0.18, 0.0, 1.0);
    float l2 = luma(col);
    col *= mix(uShadowTint, vec3(1.0), smoothstep(0.0, 0.45, l2));
    col = mix(col, col * uHighTint, smoothstep(0.35, 1.0, l2));

    // vignette
    vec2 vv = cen * vec2(1.0, uOutRes.y / uOutRes.x) * 2.0;
    col *= mix(1.0, smoothstep(1.35, 0.25, length(vv * vec2(1.0, 1.9))), uVignette);

    col = mix(col, uFlashCol, clamp(uFlash, 0.0, 1.0));
    col = mix(col, vec3(0.0), clamp(uFade, 0.0, 1.0));

    // text (white, premultiplied) with a soft glow
    if (uTextAlpha > 0.0) {
        float a = texture(uText, tuv).a;
        float glow = textureLod(uText, tuv, 4.0).a * 0.5 + textureLod(uText, tuv, 6.0).a * 0.35;
        col = mix(col, vec3(0.93, 0.9, 0.84), a * uTextAlpha);
        col += vec3(1.0, 0.8, 0.55) * glow * uTextAlpha * 0.25;
    }

    // sRGB
    col = mix(col * 12.92, 1.055 * pow(col, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, col));
    // film grain (luma dependent) + dither
    float n = h12(frag + fract(uFrame * 0.6180339) * 1000.0) + h12(frag * 1.37 + fract(uFrame * 0.41421) * 777.0) - 1.0;
    col += n * uGrain * (0.35 + 0.65 * (1.0 - abs(luma(col) - 0.5) * 2.0));
    col += (h12(frag + uFrame * 3.1) - 0.5) / 255.0;
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
#endif
