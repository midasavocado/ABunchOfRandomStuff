// Export the lyric/melody table (syllables with absolute times) from index.html to JSON for sing.py.
// Usage: node export_lines.js lines.json
const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
const path = require('path');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://' + path.resolve(__dirname, '..', 'index.html'));
  const lines = await p.evaluate(() => window.SPARE_PARTS.LINES.map(l => ({ bar: l.bar, harm: l.harm || null, shout: !!l.shout, toks: l.toks })));
  require('fs').writeFileSync(process.argv[2] || 'lines.json', JSON.stringify(lines, null, 1));
  await b.close();
})();
