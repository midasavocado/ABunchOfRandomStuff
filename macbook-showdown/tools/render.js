// Frame-exact renderer for index.html (headless Chromium via Playwright).
//
//   node render.js events OUT.json                 sound-effect and music events placed by the page
//   node render.js stills OUTDIR t1 t2 ...         PNG stills at the given times (seconds)
//   node render.js video OUT.mp4 FPS FROM TO       frames FROM..TO-1 piped into ffmpeg (H.264)
//   node render.js all OUT.mp4 FPS WORKERS         the whole video, split across workers, then joined
//
// FFMPEG env var points at an ffmpeg binary (default: ffmpeg on PATH).
const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
const fs = require('fs'), path = require('path'), { spawn } = require('child_process');
const PAGE = 'file://' + path.resolve(__dirname, '..', 'index.html') + '?render=1';
const FFMPEG = process.env.FFMPEG || 'ffmpeg';

async function open() {
  const b = await chromium.launch({ args: ['--disable-gpu-vsync', '--force-color-profile=srgb', '--font-render-hinting=none', '--allow-file-access-from-files', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  const errs = [];
  p.on('pageerror', e => errs.push(e.message));
  p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  p.on('requestfailed', r => errs.push('failed: ' + r.url()));
  await p.goto(PAGE);
  await p.evaluate(() => window.ready);
  if (errs.length) console.error('page errors:', errs);
  return { b, p };
}

function run(cmd, args) {
  return new Promise((res, rej) => { const c = spawn(cmd, args, { stdio: 'inherit' }); c.on('exit', code => code ? rej(new Error(cmd + ' exited ' + code)) : res()); });
}

async function video(out, fps, from, to, tag = '') {
  const { b, p } = await open();
  const ff = spawn(FFMPEG, ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'png', '-i', '-',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '14', '-tune', 'animation', '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', out], { stdio: ['pipe', 'inherit', 'inherit'] });
  const done = new Promise((res, rej) => ff.on('exit', c => c ? rej(new Error('ffmpeg ' + c)) : res()));
  await p.evaluate(() => renderAt(0));
  // Page.captureScreenshot with optimizeForSpeed is still lossless PNG, ~5x faster than page.screenshot
  const cdp = await p.context().newCDPSession(p);
  const t0 = Date.now();
  for (let f = from; f < to; f++) {
    await p.evaluate(t => renderAt(t), f / fps);
    const { data } = await cdp.send('Page.captureScreenshot', { format: 'png', optimizeForSpeed: true });
    const png = Buffer.from(data, 'base64');
    if (!ff.stdin.write(png)) await new Promise(r => ff.stdin.once('drain', r));
    if ((f - from) % 150 === 0) console.log(`${tag} frame ${f}/${to} (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  }
  ff.stdin.end();
  await done;
  await b.close();
}

(async () => {
  const [mode, out, ...rest] = process.argv.slice(2);
  if (mode === 'events') {
    const { b, p } = await open();
    fs.writeFileSync(out, JSON.stringify(await p.evaluate(() => getEvents()), null, 1));
    await b.close();
  } else if (mode === 'stills') {
    const { b, p } = await open();
    fs.mkdirSync(out, { recursive: true });
    for (const t of rest) {
      await p.evaluate(t => renderAt(t), +t);
      await p.screenshot({ path: path.join(out, `t${(+t).toFixed(2).padStart(7, '0')}.png`) });
    }
    await b.close();
  } else if (mode === 'video') {
    const [fps, from, to] = rest.map(Number);
    await video(out, fps, from, to);
  } else if (mode === 'all') {
    const [fps, workers] = rest.map(Number);
    const { b, p } = await open();
    const total = await p.evaluate(() => getEvents().total);
    await b.close();
    const frames = Math.round(total * fps), per = Math.ceil(frames / workers), parts = [];
    const jobs = [];
    for (let w = 0; w < workers; w++) {
      const a = w * per, z = Math.min(frames, a + per), part = `${out}.part${w}.mp4`;
      parts.push(part);
      jobs.push(video(part, fps, a, z, `[w${w}]`));
    }
    await Promise.all(jobs);
    const list = `${out}.parts.txt`;
    fs.writeFileSync(list, parts.map(f => `file '${path.resolve(f)}'`).join('\n'));
    await run(FFMPEG, ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy', out]);
    parts.forEach(f => fs.unlinkSync(f)); fs.unlinkSync(list);
  }
})().catch(e => { console.error(e); process.exit(1); });
