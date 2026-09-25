// Render the page's own audio in headless Chromium (OfflineAudioContext, sample-exact) for testing and
// for the video. Usage: node render_page.js OUTDIR [--fonts DIR]
// Writes mix.wav, vocal.wav (vocal only), band.wav (no vocal), drums.wav, plucks.json, vox.json.
const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
const fs = require('fs'), path = require('path');
(async () => {
  const out = process.argv[2];
  const b = await chromium.launch();
  const p = await b.newPage();
  const errs = [];
  p.on('pageerror', e => errs.push(e.message));
  if (process.env.FONT_ROUTES) await require(process.env.FONT_ROUTES)(p);
  await p.goto('file://' + path.resolve(__dirname, '..', 'index.html'));
  const vox = await p.evaluate(async () => { const v = await window.SPARE_PARTS.vocalsReady; return { state: v.state, offset: v.offset }; });
  fs.writeFileSync(path.join(out, 'vox.json'), JSON.stringify(vox));
  const stems = { mix: {}, vocal: { gtr: 1, bass: 1, drums: 1, fx: 1 }, band: { voice: 1 }, drums: { gtr: 1, bass: 1, fx: 1, voice: 1, onlyKick: 1 } };
  for (const [k, m] of Object.entries(stems)) {
    const b64 = await p.evaluate(m => window.SPARE_PARTS.renderWav(m), m);
    fs.writeFileSync(path.join(out, k + '.wav'), Buffer.from(b64, 'base64'));
  }
  const notes = [40, 43, 45, 47, 48, 50, 52, 55, 57, 59, 60, 61, 62, 64, 66, 67];
  const plucks = {};
  for (const m of notes) plucks[m] = await p.evaluate(m => window.SPARE_PARTS.renderPluck(m), m);
  fs.writeFileSync(path.join(out, 'plucks.json'), JSON.stringify(plucks));
  console.log(JSON.stringify({ vox, errors: errs }));
  await b.close();
})();
