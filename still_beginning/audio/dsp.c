// STILL BEGINNING - small DSP kernels called from numpy via ctypes.
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.14159265358979323846

// Zavalishin TPT state-variable filter. mode 0=LP 1=BP 2=HP 3=notch. cutoff per-sample (Hz).
void svf(const float *in, float *out, int n, const float *fc, float q, int mode, float sr) {
    double ic1 = 0, ic2 = 0;
    for (int i = 0; i < n; i++) {
        double f = fc[i];
        if (f < 10) f = 10;
        if (f > sr * 0.49) f = sr * 0.49;
        double g = tan(PI * f / sr);
        double k = 1.0 / q;
        double a1 = 1.0 / (1.0 + g * (g + k));
        double a2 = g * a1, a3 = g * a2;
        double v0 = in[i];
        double v3 = v0 - ic2;
        double v1 = a1 * ic1 + a2 * v3;
        double v2 = ic2 + a2 * ic1 + a3 * v3;
        ic1 = 2 * v1 - ic1;
        ic2 = 2 * v2 - ic2;
        double y;
        if (mode == 0) y = v2;
        else if (mode == 1) y = v1;
        else if (mode == 2) y = v0 - k * v1 - v2;
        else y = v0 - k * v1;
        out[i] = (float)y;
    }
}

// 4-pole ladder, one-sample feedback, tanh saturation on each stage (2x internal oversampling).
void ladder(const float *in, float *out, int n, const float *fc, float res, float drive, float sr) {
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0, y3 = 0, prev = 0;
    double osr = sr * 2;
    for (int i = 0; i < n; i++) {
        double f = fc[i];
        if (f < 10) f = 10;
        if (f > sr * 0.45) f = sr * 0.45;
        double g = 1 - exp(-2 * PI * f / osr);
        double k = 4.0 * res;
        double x = in[i];
        double acc = 0;
        for (int o = 0; o < 2; o++) {
            double xi = o == 0 ? 0.5 * (prev + x) : x;
            double u = tanh(drive * (xi - k * (y3 - 0.5 * xi * 0))) ;
            s0 += g * (u - tanh(s0));
            s1 += g * (tanh(s0) - tanh(s1));
            s2 += g * (tanh(s1) - tanh(s2));
            s3 += g * (tanh(s2) - tanh(s3));
            y3 = s3;
            acc += s3;
        }
        prev = x;
        out[i] = (float)(0.5 * acc * (1 + res * 1.2) / (drive > 0 ? drive : 1));
    }
}

// one-pole attack/release envelope follower on |x|
void envfollow(const float *in, float *out, int n, float att_ms, float rel_ms, float sr) {
    double a = exp(-1.0 / (att_ms * 0.001 * sr));
    double r = exp(-1.0 / (rel_ms * 0.001 * sr));
    double e = 0;
    for (int i = 0; i < n; i++) {
        double x = fabs(in[i]);
        if (x > e) e = a * e + (1 - a) * x;
        else e = r * e + (1 - r) * x;
        out[i] = (float)e;
    }
}

// smoothed gain computer: out = smoothed gain for a peak-limited signal (lookahead handled in numpy)
void gainsmooth(const float *target, float *out, int n, float att_ms, float rel_ms, float sr) {
    double a = exp(-1.0 / (att_ms * 0.001 * sr));
    double r = exp(-1.0 / (rel_ms * 0.001 * sr));
    double g = 1.0;
    for (int i = 0; i < n; i++) {
        double t = target[i];
        if (t < g) g = a * g + (1 - a) * t;
        else g = r * g + (1 - r) * t;
        out[i] = (float)g;
    }
}

// 8-line feedback delay network reverb with modulated delays and damping.
static double hash01(unsigned int x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16;
    return (x & 0xffffff) / (double)0xffffff;
}

void fdn_reverb(const float *inL, const float *inR, float *outL, float *outR, int n,
                float size, float decay_s, float damp_hz, float predelay_ms, float mod_depth, float sr) {
    const int N = 8;
    static const double base[8] = {1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617};
    int len[8];
    double *buf[8];
    int maxlen = 0;
    for (int j = 0; j < N; j++) {
        len[j] = (int)(base[j] * size * sr / 44100.0 * 2.2);
        if (len[j] + 64 > maxlen) maxlen = len[j] + 64;
    }
    for (int j = 0; j < N; j++) { buf[j] = calloc(maxlen, sizeof(double)); }
    int pd = (int)(predelay_ms * 0.001 * sr);
    int pdlen = pd + 1;
    double *pdL = calloc(pdlen, sizeof(double)), *pdR = calloc(pdlen, sizeof(double));
    int pdw = 0;
    double gfb[8], lpz[8], lpa;
    lpa = exp(-2 * PI * damp_hz / sr);
    for (int j = 0; j < N; j++) {
        gfb[j] = pow(10.0, -3.0 * len[j] / (decay_s * sr));
        lpz[j] = 0;
    }
    // input diffusion allpasses
    const int NA = 4;
    static const double apb[4] = {142, 107, 379, 277};
    int aplen[4]; double *apbufL[4], *apbufR[4]; int apw[4];
    for (int a = 0; a < NA; a++) {
        aplen[a] = (int)(apb[a] * sr / 44100.0 * size);
        if (aplen[a] < 2) aplen[a] = 2;
        apbufL[a] = calloc(aplen[a], sizeof(double)); apbufR[a] = calloc(aplen[a], sizeof(double)); apw[a] = 0;
    }
    int w = 0;
    double ph[8];
    for (int j = 0; j < N; j++) ph[j] = hash01(j * 7 + 3) * 2 * PI;
    for (int i = 0; i < n; i++) {
        // predelay
        double xl = pdL[pdw], xr = pdR[pdw];
        pdL[pdw] = inL[i]; pdR[pdw] = inR[i];
        pdw = (pdw + 1) % pdlen;
        if (pd == 0) { xl = inL[i]; xr = inR[i]; }
        // diffusion
        for (int a = 0; a < NA; a++) {
            double g = 0.6;
            double bl = apbufL[a][apw[a]], br = apbufR[a][apw[a]];
            double vl = xl + g * bl, vr = xr + g * br;
            apbufL[a][apw[a]] = vl; apbufR[a][apw[a]] = vr;
            xl = bl - g * vl; xr = br - g * vr;
            apw[a] = (apw[a] + 1) % aplen[a];
        }
        double o[8];
        for (int j = 0; j < N; j++) {
            double m = mod_depth * sr * 0.001 * sin(ph[j] + 2 * PI * (0.1 + 0.07 * j) * i / sr);
            double d = len[j] + m;
            double rp = w - d;
            while (rp < 0) rp += maxlen;
            int i0 = (int)rp; double fr = rp - i0;
            int i1 = (i0 + 1) % maxlen;
            double v = buf[j][i0] * (1 - fr) + buf[j][i1] * fr;
            lpz[j] = v * (1 - lpa) + lpz[j] * lpa;
            o[j] = lpz[j] * gfb[j];
        }
        // Hadamard 8x8 (fast)
        double h[8];
        for (int j = 0; j < 8; j++) h[j] = o[j];
        for (int s = 1; s < 8; s <<= 1)
            for (int j = 0; j < 8; j += 2 * s)
                for (int k = j; k < j + s; k++) {
                    double a = h[k], b = h[k + s];
                    h[k] = a + b; h[k + s] = a - b;
                }
        double sc = 1.0 / sqrt(8.0);
        for (int j = 0; j < N; j++) {
            double inj = (j & 1) ? xr : xl;
            buf[j][w] = h[j] * sc + inj * 0.5;
        }
        outL[i] = (float)(o[0] + o[2] + o[4] + o[6]) * 0.5f;
        outR[i] = (float)(o[1] + o[3] + o[5] + o[7]) * 0.5f;
        w = (w + 1) % maxlen;
    }
    for (int j = 0; j < N; j++) free(buf[j]);
    for (int a = 0; a < NA; a++) { free(apbufL[a]); free(apbufR[a]); }
    free(pdL); free(pdR);
}

// ---------------------------------------------------------------------------------------------
// Extensions for the score (compose.py)
// ---------------------------------------------------------------------------------------------

static inline double polyblep(double t, double dt) {
    if (t < dt) { t /= dt; return t + t - t * t - 1.0; }
    if (t > 1.0 - dt) { t = (t - 1.0) / dt; return t * t + t + t + 1.0; }
    return 0.0;
}

// Band-limited oscillator with per-sample frequency. shape 0 = saw, 1 = pulse (width pw), 2 = triangle
// (leaky-integrated polyBLEP square). Returns final phase so notes can be continued.
double osc_blep(float *out, int n, const float *freq, double phase, int shape, float pw, float sr) {
    double tri = 0.0;
    for (int i = 0; i < n; i++) {
        double dt = freq[i] / sr;
        if (dt < 0) dt = 0;
        if (dt > 0.45) dt = 0.45;
        double y;
        if (shape == 0) {
            y = 2.0 * phase - 1.0;
            y -= polyblep(phase, dt);
        } else {
            y = phase < pw ? 1.0 : -1.0;
            y += polyblep(phase, dt);
            double p2 = phase - pw; if (p2 < 0) p2 += 1.0;
            y -= polyblep(p2, dt);
            if (shape == 2) {
                tri = dt * 4.0 * y + (1.0 - dt * 0.05) * tri;
                y = tri;
            }
        }
        out[i] = (float)y;
        phase += dt;
        if (phase >= 1.0) phase -= 1.0;
    }
    return phase;
}

// Stereo feedback delay with ping-pong cross feed and band-limited feedback path.
// dL, dR in samples (fractional ok). fb = feedback gain. cross = 0..1 amount of ping-pong.
void delay_pp(const float *inL, const float *inR, float *outL, float *outR, int n,
              float dL, float dR, float fb, float lp_hz, float hp_hz, float cross, float sr) {
    int maxlen = (int)(fmax(dL, dR) + 8);
    double *bl = calloc(maxlen, sizeof(double)), *br = calloc(maxlen, sizeof(double));
    double lpa = exp(-2 * PI * lp_hz / sr), hpa = exp(-2 * PI * hp_hz / sr);
    double lzl = 0, lzr = 0, hzl = 0, hzr = 0;
    int w = 0;
    for (int i = 0; i < n; i++) {
        double rpl = w - dL; while (rpl < 0) rpl += maxlen;
        double rpr = w - dR; while (rpr < 0) rpr += maxlen;
        int il = (int)rpl, ir = (int)rpr; double fl = rpl - il, fr = rpr - ir;
        double yl = bl[il] * (1 - fl) + bl[(il + 1) % maxlen] * fl;
        double yr = br[ir] * (1 - fr) + br[(ir + 1) % maxlen] * fr;
        // filter the wet signal (tone of repeats)
        lzl = yl * (1 - lpa) + lzl * lpa; lzr = yr * (1 - lpa) + lzr * lpa;
        hzl = lzl * (1 - hpa) + hzl * hpa; hzr = lzr * (1 - hpa) + hzr * hpa;
        double wl = lzl - hzl, wr = lzr - hzr;
        outL[i] = (float)wl; outR[i] = (float)wr;
        double fL = (1 - cross) * wl + cross * wr, fR = (1 - cross) * wr + cross * wl;
        bl[w] = inL[i] + fb * fL;
        br[w] = inR[i] + fb * fR;
        w = (w + 1) % maxlen;
    }
    free(bl); free(br);
}

// Stereo chorus/ensemble: 3 modulated taps per side with phase-offset LFOs (wet only).
void chorus(const float *inL, const float *inR, float *outL, float *outR, int n,
            float base_ms, float depth_ms, float rate_hz, float sr) {
    int maxlen = (int)((base_ms + depth_ms * 2 + 5) * 0.001 * sr) + 8;
    double *bl = calloc(maxlen, sizeof(double)), *br = calloc(maxlen, sizeof(double));
    int w = 0;
    for (int i = 0; i < n; i++) {
        bl[w] = inL[i]; br[w] = inR[i];
        double accL = 0, accR = 0;
        for (int v = 0; v < 3; v++) {
            double ph = 2 * PI * (rate_hz * (1 + 0.13 * v) * i / sr) + v * 2.0944;
            double dl = (base_ms + depth_ms * (1 + sin(ph))) * 0.001 * sr;
            double dr = (base_ms + depth_ms * (1 + sin(ph + 1.5708))) * 0.001 * sr;
            double rp = w - dl; while (rp < 0) rp += maxlen;
            int i0 = (int)rp; double f = rp - i0;
            accL += bl[i0] * (1 - f) + bl[(i0 + 1) % maxlen] * f;
            rp = w - dr; while (rp < 0) rp += maxlen;
            i0 = (int)rp; f = rp - i0;
            accR += br[i0] * (1 - f) + br[(i0 + 1) % maxlen] * f;
        }
        outL[i] = (float)(accL / 3.0); outR[i] = (float)(accR / 3.0);
        w = (w + 1) % maxlen;
    }
    free(bl); free(br);
}

// Stereo-linked feed-forward compressor (log domain, soft knee). Writes gain (linear) to g.
void comp_gain(const float *detL, const float *detR, float *g, int n, float thr_db, float ratio,
               float att_ms, float rel_ms, float knee_db, float sr) {
    double a = exp(-1.0 / (att_ms * 0.001 * sr));
    double r = exp(-1.0 / (rel_ms * 0.001 * sr));
    double env = 0; // gain reduction in dB (positive)
    for (int i = 0; i < n; i++) {
        double x = fmax(fabs(detL[i]), fabs(detR[i]));
        double xdb = 20 * log10(x + 1e-9);
        double over = xdb - thr_db;
        double gr;
        if (over <= -knee_db / 2) gr = 0;
        else if (over >= knee_db / 2) gr = over * (1 - 1 / ratio);
        else { double t = over + knee_db / 2; gr = (1 - 1 / ratio) * t * t / (2 * knee_db); }
        if (gr > env) env = a * env + (1 - a) * gr;
        else env = r * env + (1 - r) * gr;
        g[i] = (float)pow(10.0, -env / 20.0);
    }
}

// Karplus-Strong plucked string with fractional-delay allpass tuning, one-zero/one-pole loss filter.
void karplus(const float *exc, float *out, int n, float f0, float t60, float bright, float sr) {
    double period = sr / f0;
    int N = (int)floor(period - 0.5);
    if (N < 2) N = 2;
    double frac = period - N - 0.5;  // loop filter adds ~0.5 sample delay
    double c = (1 - frac) / (1 + frac); // allpass coefficient
    double g = pow(10.0, -3.0 / (t60 * f0));
    double *buf = calloc(N + 2, sizeof(double));
    int w = 0; double lp = 0, apx = 0, apy = 0;
    for (int i = 0; i < n; i++) {
        double y = buf[w];
        // loss filter: blend of one-pole lowpass
        lp = bright * y + (1 - bright) * lp;
        double v = g * lp;
        // allpass fractional delay
        double ap = c * v + apx - c * apy;
        apx = v; apy = ap;
        buf[w] = exc[i] + ap;
        out[i] = (float)y;
        w = (w + 1) % N;
    }
    free(buf);
}
