// Renders index.html to an MP4: every frame is drawn by the page in headless Chromium,
// and the soundtrack is the page's own OfflineAudioContext render.
//
//   FFMPEG=/path/to/ffmpeg node tools/render.js [out.mp4] [--stills 0,4.5,12]
//
// --stills writes PNGs of the given times instead of a video (for checking frames).
// --audio writes only the soundtrack, as out.wav.
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const FFMPEG = process.env.FFMPEG || 'ffmpeg';
const args = process.argv.slice(2);
const stillsAt = args.includes('--stills') ? args[args.indexOf('--stills') + 1] : null;
const out = path.resolve(args.find(a => a.endsWith('.mp4')) ||
  path.join(__dirname, '..', 'made-of-code.mp4'));
const page_url = 'file://' + path.resolve(__dirname, '..', 'index.html') + '?render';

function wav(chans, sr) {
  const n = chans[0].length, b = Buffer.alloc(44 + n * 4);
  b.write('RIFF', 0); b.writeUInt32LE(36 + n * 4, 4); b.write('WAVEfmt ', 8);
  b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(2, 22);
  b.writeUInt32LE(sr, 24); b.writeUInt32LE(sr * 4, 28); b.writeUInt16LE(4, 32);
  b.writeUInt16LE(16, 34); b.write('data', 36); b.writeUInt32LE(n * 4, 40);
  for (let i = 0; i < n; i++) for (let c = 0; c < 2; c++) {
    const v = Math.max(-1, Math.min(1, chans[c][i]));
    b.writeInt16LE(Math.round(v * 32767), 44 + i * 4 + c * 2);
  }
  return b;
}

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROME || undefined,
    args: ['--autoplay-policy=no-user-gesture-required'],
  });
  const page = await browser.newPage({ viewport: { width: 540, height: 960 } });
  page.on('console', m => console.log('[page]', m.text()));
  page.on('pageerror', e => { console.error('[page error]', e); process.exit(1); });
  await page.goto(page_url);
  const info = await page.evaluate(() => window.prepare());
  console.log('prepared', info);
  const { W, H, FPS, LEN } = await page.evaluate(() => ({ W, H, FPS, LEN }));

  const grab = async t => Buffer.from((await page.evaluate(t => {
    frame(t); return document.getElementById('c').toDataURL('image/png');
  }, t)).split(',')[1], 'base64');

  if (stillsAt) {
    for (const t of stillsAt.split(',').map(Number)) {
      const f = path.join(path.dirname(out), `still_${String(t).padStart(5, '0')}.png`);
      fs.writeFileSync(f, await grab(t)); console.log('wrote', f);
    }
    await browser.close(); return;
  }

  // soundtrack
  const audio = await page.evaluate(() => {
    const m = music(), enc = d => {
      const q = new Int16Array(d.length);
      for (let i = 0; i < d.length; i++) q[i] = Math.max(-32767, Math.min(32767, d[i] * 32767));
      let s = ''; const u = new Uint8Array(q.buffer);
      for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode(...u.subarray(i, i + 0x8000));
      return btoa(s);
    };
    return { sr: m.sampleRate, ch: [enc(m.getChannelData(0)), enc(m.getChannelData(1))] };
  });
  const chans = audio.ch.map(b64 => {
    const b = Buffer.from(b64, 'base64'), q = new Int16Array(b.buffer, b.byteOffset, b.length / 2);
    return Float32Array.from(q, v => v / 32767);
  });
  const wavPath = out.replace(/\.mp4$/, '.wav');
  fs.writeFileSync(wavPath, wav(chans, audio.sr));
  if (args.includes('--audio')) { await browser.close(); console.log('wrote', wavPath); return; }

  // picture
  const frames = Math.round(LEN * FPS);
  const ff = spawn(FFMPEG, ['-y', '-loglevel', 'error',
    '-f', 'image2pipe', '-framerate', String(FPS), '-i', '-',
    '-i', wavPath,
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-maxrate', '10M', '-bufsize', '20M',
    '-pix_fmt', 'yuv420p',
    '-tune', 'animation', '-movflags', '+faststart',
    '-c:a', 'aac', '-b:a', '192k', '-shortest', out], { stdio: ['pipe', 'inherit', 'inherit'] });
  const t0 = Date.now();
  for (let i = 0; i < frames; i++) {
    const png = await grab(i / FPS);
    if (!ff.stdin.write(png)) await new Promise(r => ff.stdin.once('drain', r));
    if (i % 60 === 0) console.log(`frame ${i}/${frames}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
  fs.unlinkSync(wavPath);
  await browser.close();
  console.log('wrote', out, W + 'x' + H);
})();
