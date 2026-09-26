// The 3D layer: two MacBook Pros built from real dimensions, on a transparent canvas.
// Bundled to assets/js/stage.js (npm run bundle). The page drives everything through window.STAGE.
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';

const W = 1920, H = 1080;

// rounded-rectangle slab: w (x) × d (z) × h (y), plan radius r, edge bevel b; bottom at y = 0
function slab(w, d, h, r, b, seg = 10) {
  const s = new THREE.Shape(), x = -w / 2 + b, y = -d / 2 + b, ww = w - 2 * b, dd = d - 2 * b, rr = Math.max(0.001, r - b);
  s.moveTo(x + rr, y); s.lineTo(x + ww - rr, y); s.quadraticCurveTo(x + ww, y, x + ww, y + rr);
  s.lineTo(x + ww, y + dd - rr); s.quadraticCurveTo(x + ww, y + dd, x + ww - rr, y + dd);
  s.lineTo(x + rr, y + dd); s.quadraticCurveTo(x, y + dd, x, y + dd - rr);
  s.lineTo(x, y + rr); s.quadraticCurveTo(x, y, x + rr, y);
  const g = new THREE.ExtrudeGeometry(s, { depth: h - 2 * b, bevelEnabled: b > 0, bevelThickness: b, bevelSize: b, bevelSegments: 4, curveSegments: seg });
  g.rotateX(-Math.PI / 2);
  g.translate(0, b, 0);
  g.computeVertexNormals();
  return g;
}
function roundRectShape(w, d, r) {
  const s = new THREE.Shape(), x = -w / 2, y = -d / 2;
  s.moveTo(x + r, y); s.lineTo(x + w - r, y); s.quadraticCurveTo(x + w, y, x + w, y + r);
  s.lineTo(x + w, y + d - r); s.quadraticCurveTo(x + w, y + d, x + w - r, y + d);
  s.lineTo(x + r, y + d); s.quadraticCurveTo(x, y + d, x, y + d - r);
  s.lineTo(x, y + r); s.quadraticCurveTo(x, y, x + r, y);
  return s;
}
// flat rounded rectangle lying in the xz plane, facing +y
function flatRR(w, d, r) {
  const g = new THREE.ShapeGeometry(roundRectShape(w, d, r), 8);
  // ShapeGeometry UVs are in shape units; normalise them to 0..1 so textures map across the shape
  const uv = g.attributes.uv, pos = g.attributes.position;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, (pos.getX(i) + w / 2) / w, (pos.getY(i) + d / 2) / d);
  g.rotateX(-Math.PI / 2);
  return g;
}
function canvasTex(w, h, srgb = true) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });   // CPU-backed: cheap to upload every frame
  const t = new THREE.CanvasTexture(c);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.generateMipmaps = false; t.minFilter = THREE.LinearFilter;
  return { canvas: c, ctx, tex: t };
}
function dotTexture(cols, rows, bg = 'rgba(0,0,0,0)') {
  const { canvas, ctx, tex } = canvasTex(cols * 8, rows * 8, false);
  ctx.fillStyle = bg; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#fff';
  for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++) { ctx.beginPath(); ctx.arc(i * 8 + 4 + (j % 2) * 4, j * 8 + 4, 2.1, 0, 7); ctx.fill(); }
  tex.needsUpdate = true;
  return tex;
}

export function createStage(canvas, opts = {}) {
  // Only the laptops are 3D. The canvas is transparent; backgrounds, glows and text are 2D layers
  // in the page. No post-processing, so a frame is cheap even without a GPU.
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: opts.aa ?? true, alpha: true, premultipliedAlpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1);
  renderer.setSize(W, H, false);
  renderer.setClearColor(0x000000, 0);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.9;

  const camera = new THREE.PerspectiveCamera(28, W / H, 0.05, 200);
  const cam = { pos: new THREE.Vector3(0, 2.4, 11), target: new THREE.Vector3(0, 0.9, 0), roll: 0, fov: 28 };

  const hemi = new THREE.HemisphereLight('#c8d0ff', '#1a1020', 0.35); scene.add(hemi);
  const key = new THREE.SpotLight('#fff4e6', 160, 40, 0.5, 0.8, 1.2); key.position.set(3, 9, 7); scene.add(key); scene.add(key.target);
  const rimL = new THREE.PointLight('#ff2d55', 0, 30, 1.4); rimL.position.set(-6, 3.5, -3); scene.add(rimL);
  const rimR = new THREE.PointLight('#2f7bff', 0, 30, 1.4); rimR.position.set(6, 3.5, -3); scene.add(rimR);
  const top = new THREE.SpotLight('#ffffff', 0, 30, 0.35, 0.6, 1.2); top.position.set(0, 10, 1); scene.add(top); scene.add(top.target);

  const laptops = { '13': makeLaptop('13'), '14': makeLaptop('14') };
  laptops['13'].group.position.set(-2.1, 0, 0);
  laptops['14'].group.position.set(2.1, 0, 0);
  scene.add(laptops['13'].group, laptops['14'].group);

  // sprites (Fluent 3D emoji etc.) living in the scene
  const texCache = {};
  const loader = new THREE.TextureLoader();
  function spriteTex(url) {
    if (!texCache[url]) { const t = loader.load(url); t.colorSpace = THREE.SRGBColorSpace; texCache[url] = t; }
    return texCache[url];
  }
  function sprite(url, size = 1) {
    const m = new THREE.SpriteMaterial({ map: typeof url === 'string' ? spriteTex(url) : url, transparent: true, depthWrite: false, toneMapped: false });
    const s = new THREE.Sprite(m); s.scale.set(size, size, 1); scene.add(s);
    return s;
  }
  // flat textured card in 3D (e.g. a dongle, a label)
  function card(url, w, h) {
    const m = new THREE.MeshBasicMaterial({ map: spriteTex(url), transparent: true, side: THREE.DoubleSide, depthWrite: false });
    const p = new THREE.Mesh(new THREE.PlaneGeometry(w, h), m); scene.add(p); return p;
  }

  const v = new THREE.Vector3();
  function project(obj, offset) {
    obj.updateWorldMatrix(true, false);
    v.set(0, 0, 0); if (offset) v.copy(offset);
    obj.localToWorld(v); v.project(camera);
    return { x: (v.x + 1) / 2 * W, y: (1 - v.y) / 2 * H, behind: v.z > 1 };
  }
  function projectPoint(p) { v.copy(p).project(camera); return { x: (v.x + 1) / 2 * W, y: (1 - v.y) / 2 * H }; }

  const drift = new THREE.Vector3();
  function prepare(t = 0) {
    // a slow handheld float so no shot is ever perfectly static
    drift.set(Math.sin(t * 0.61) * 0.05, Math.sin(t * 0.83 + 1) * 0.03, Math.sin(t * 0.47 + 2) * 0.04).multiplyScalar(cam.float ?? 1);
    camera.position.copy(cam.pos).add(drift);
    camera.fov = cam.fov; camera.updateProjectionMatrix();
    camera.up.set(Math.sin(cam.roll), Math.cos(cam.roll), 0);
    camera.lookAt(cam.target);
    for (const k in laptops) laptops[k].update();
    camera.updateMatrixWorld();
  }
  function render(t = 0) { prepare(t); renderer.render(scene, camera); }

  return { THREE, renderer, scene, camera, cam, key, rimL, rimR, top, hemi, laptops, sprite, card, project, projectPoint, prepare, render, spriteTex };
}

// ---------------------------------------------------------------------------------------------
// The laptops
// ---------------------------------------------------------------------------------------------
// Matcap materials: studio lighting baked into a small sphere image, so shading costs one texture
// lookup per pixel (no lights, no environment maps). Each matcap is painted procedurally.
const MATCAPS = {};
function matcap(color, gloss = 0.6, rim = 0.5) {
  const key = color + gloss + rim;
  if (MATCAPS[key]) return MATCAPS[key];
  const S = 256, c = document.createElement('canvas'); c.width = c.height = S;
  const x = c.getContext('2d'), col = new THREE.Color(color);
  const rgb = (k, a = 1) => `rgba(${Math.min(255, col.r * 255 * k) | 0},${Math.min(255, col.g * 255 * k) | 0},${Math.min(255, col.b * 255 * k) | 0},${a})`;
  let g = x.createLinearGradient(0, 0, 0, S); g.addColorStop(0, rgb(1.35)); g.addColorStop(0.45, rgb(1.0)); g.addColorStop(1, rgb(0.42));
  x.fillStyle = g; x.fillRect(0, 0, S, S);
  // softbox reflections: a broad key highlight and a thin horizon band
  g = x.createRadialGradient(S * 0.36, S * 0.28, 0, S * 0.36, S * 0.28, S * 0.45); g.addColorStop(0, `rgba(255,255,255,${0.75 * gloss})`); g.addColorStop(1, 'rgba(255,255,255,0)');
  x.fillStyle = g; x.fillRect(0, 0, S, S);
  g = x.createLinearGradient(0, S * 0.5, 0, S * 0.62); g.addColorStop(0, 'rgba(255,255,255,0)'); g.addColorStop(0.5, `rgba(255,255,255,${0.35 * gloss})`); g.addColorStop(1, 'rgba(255,255,255,0)');
  x.fillStyle = g; x.fillRect(0, 0, S, S);
  // rim light and edge falloff
  g = x.createRadialGradient(S / 2, S / 2, S * 0.3, S / 2, S / 2, S * 0.5); g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(0.8, 'rgba(0,0,0,0.25)'); g.addColorStop(0.97, `rgba(255,255,255,${0.45 * rim})`); g.addColorStop(1, 'rgba(255,255,255,0)');
  x.fillStyle = g; x.fillRect(0, 0, S, S);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  return (MATCAPS[key] = t);
}
const Mat = o => new THREE.MeshMatcapMaterial({ matcap: matcap(o.color, o.gloss ?? (1 - (o.roughness ?? 0.4)), o.rim ?? 0.5), toneMapped: false });
function makeLaptop(kind) {
  const is14 = kind === '14';
  const Wd = is14 ? 3.126 : 3.041, D = is14 ? 2.212 : 2.124;
  const Tb = is14 ? 0.098 : 0.094, Tl = is14 ? 0.056 : 0.06;
  const alu = Mat({
    color: is14 ? '#3f4146' : '#d2d4d8', metalness: 1, roughness: is14 ? 0.45 : 0.3, clearcoat: 0.25, clearcoatRoughness: 0.4,
  });
    const keyMat = Mat({ color: '#1a1a1e', gloss: 0.25, rim: 0.2 });

  const group = new THREE.Group();            // position/rotation of the whole laptop (origin: bottom centre)
  const body = new THREE.Group(); group.add(body);
  const base = new THREE.Mesh(slab(Wd, D, Tb, is14 ? 0.12 : 0.14, is14 ? 0.02 : 0.035), alu);
  body.add(base);

  // contact shadow
  const sh = document.createElement('canvas'); sh.width = sh.height = 256;
  { const c = sh.getContext('2d'); const g = c.createRadialGradient(128, 128, 10, 128, 128, 128); g.addColorStop(0, 'rgba(0,0,0,0.9)'); g.addColorStop(0.55, 'rgba(0,0,0,0.55)'); g.addColorStop(1, 'rgba(0,0,0,0)'); c.fillStyle = g; c.fillRect(0, 0, 256, 256); }
  const shTex = new THREE.CanvasTexture(sh);
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(Wd * 1.5, D * 1.7), new THREE.MeshBasicMaterial({ map: shTex, transparent: true, depthWrite: false, opacity: 0.85 }));
  shadow.rotation.x = -Math.PI / 2; shadow.position.y = 0.004; shadow.renderOrder = 2;
  group.add(shadow);

  // keyboard
  const u = 0.19, kz0 = -D / 2 + (is14 ? 0.24 : 0.27);
  const rows = [
    is14 ? [1.45, ...Array(12).fill(1), 1] : [1, 12.45, 1],
    [...Array(13).fill(1), 1.45],
    [1.45, ...Array(12).fill(1), 1],
    [1.75, ...Array(11).fill(1), 1.7],
    [2.25, ...Array(10).fill(1), 2.2],
    [1, 1, 1, 1.25, 5.0, 1.25, 1, 1, 1, 1],
  ];
  const keyGeo = new RoundedBoxGeometry(1, 1, 1, 2, 0.18);
  const keys = new THREE.InstancedMesh(keyGeo, keyMat, 90);
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), sc = new THREE.Vector3(), p = new THREE.Vector3();
  let ki = 0, touchBar = null, tbAnchor = new THREE.Object3D();
  const legends = [];
  const LAB = [
    is14 ? ['esc', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', ''] : ['esc', '', ''],
    ['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 'delete'],
    ['tab', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '[', ']', '\\'],
    ['caps lock', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', "'", 'return'],
    ['shift', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/', 'shift'],
    ['fn', 'control', 'option', 'command', '', 'command', 'option', '◀', '▲▼', '▶'],
  ];
  const rowDepth = r => (r === 0 && !is14 ? 0.1 : 0.165);
  rows.forEach((row, r) => {
    const total = row.reduce((a, b) => a + b, 0) * u;
    let x = -total / 2;
    const z = kz0 + r * u + (r === 0 && !is14 ? 0.03 : 0);
    row.forEach((w, i) => {
      const kw = w * u - 0.028;
      if (!is14 && r === 0 && i === 1) {
        // Touch Bar: an OLED strip
        const tb = canvasTexOf(1536, 64);
        const mat = new THREE.MeshBasicMaterial({ map: tb.tex, toneMapped: false });
        const mesh = new THREE.Mesh(flatRR(kw, 0.085, 0.02), mat);
        mesh.position.set(x + w * u / 2, Tb + 0.0025, z);
        body.add(mesh);
        tbAnchor.position.copy(mesh.position); body.add(tbAnchor);
        touchBar = { ...tb, mesh, mat, w: kw };
      } else {
        const depth = r === 5 && i >= 7 ? 0.165 : rowDepth(r);
        p.set(x + w * u / 2, Tb + 0.004, z); sc.set(kw, 0.014, depth);
        if (r === 5 && i === 8) { // up/down arrows: two half-height keys
          sc.z = 0.078; p.z = z - 0.043; m4.compose(p, q, sc); keys.setMatrixAt(ki++, m4); p.z = z + 0.043;
        }
        if (r === 5 && (i === 7 || i === 9)) { sc.z = 0.078; p.z = z + 0.043; }
        m4.compose(p, q, sc); keys.setMatrixAt(ki++, m4);
        legends.push({ x: x + w * u / 2, z: r === 5 && i >= 7 ? z : p.z, w: kw, label: LAB[r][i] || '' });
      }
      x += w * u;
    });
  });
  keys.count = ki;
  body.add(keys);
  const kbW = 14.45 * u + 0.06, kbD = 6 * u + 0.04;
  {
    const lw = kbW + 0.1, ld = kbD + 0.1, cz = kz0 + 2.5 * u, PX = 1100;
    const lg = canvasTex(Math.round(lw * PX), Math.round(ld * PX));
    const c = lg.ctx; c.fillStyle = '#e2e3e8'; c.textAlign = 'center'; c.textBaseline = 'middle';
    for (const k of legends) {
      if (!k.label) continue;
      const cx = (k.x + lw / 2) * PX, cy = (k.z - cz + ld / 2) * PX, long = k.label.length > 2;
      c.font = `${long ? 600 : 500} ${long ? 30 : k.label.length === 2 && k.label[0] === 'F' ? 30 : 58}px Helvetica, Arial, sans-serif`;
      if (long && k.w > 0.3) { c.textAlign = k.x < 0 ? 'left' : 'right'; c.fillText(k.label, cx + (k.x < 0 ? -1 : 1) * (k.w * PX / 2 - 22), cy + 40); c.textAlign = 'center'; }
      else c.fillText(k.label, cx, long ? cy + 40 : cy);
    }
    lg.tex.generateMipmaps = true; lg.tex.minFilter = THREE.LinearMipmapLinearFilter; lg.tex.anisotropy = 8; lg.tex.needsUpdate = true;
    const legend = new THREE.Mesh(new THREE.PlaneGeometry(lw, ld), new THREE.MeshBasicMaterial({ map: lg.tex, transparent: true, depthWrite: false, opacity: 0.92 }));
    legend.rotation.x = -Math.PI / 2; legend.position.set(0, Tb + 0.0116, cz); legend.renderOrder = 2;
    body.add(legend);
  }
  if (is14) {  // the 14" keyboard sits in a black well
    const well = new THREE.Mesh(flatRR(kbW + 0.03, kbD + 0.02, 0.05), Mat({ color: '#101013', gloss: 0.1, rim: 0.1 }));
    well.position.set(0, Tb + 0.0008, kz0 + 2.5 * u);
    body.add(well);
  }
  // speaker grilles either side of the keyboard
  const grilleTex = dotTexture(is14 ? 7 : 4, 70);
  const grilleMat = new THREE.MeshBasicMaterial({ color: '#0a0a0c', alphaMap: grilleTex, transparent: true, depthWrite: false });
  const gw = is14 ? 0.17 : 0.1, gx = kbW / 2 + (is14 ? 0.15 : 0.1);
  const grilles = [-1, 1].map(sgn => {
    const g = new THREE.Mesh(new THREE.PlaneGeometry(gw, kbD - 0.04), grilleMat);
    g.rotation.x = -Math.PI / 2; g.position.set(sgn * gx, Tb + 0.0012, kz0 + 2.5 * u);
    body.add(g); return g;
  });
  // trackpad
  const tpW = is14 ? 1.61 : 1.6, tpD = is14 ? 0.99 : 0.99;
  const tp = new THREE.Mesh(flatRR(tpW, tpD, 0.08), Mat({ color: is14 ? '#5d5f64' : '#c9cbd0', metalness: 0.7, roughness: 0.16, clearcoat: 0.6 }));
  tp.position.set(0, Tb + 0.0008, D / 2 - 0.12 - tpD / 2);
  body.add(tp);
  // hinge
  const hinge = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, Wd - 0.7, 16), Mat({ color: '#2a2a30', gloss: 0.4 }));
  hinge.rotation.z = Math.PI / 2; hinge.position.set(0, Tb, -D / 2 + 0.035);
  body.add(hinge);

  // ports on the sides (anchors point outward, for callouts)
  const ports = {};
  const portMat = new THREE.MeshBasicMaterial({ color: '#050506', toneMapped: false });
  const ringMat = () => new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0, depthWrite: false, toneMapped: false });
  function port(name, side, z, w, h) {
    const g = new THREE.Group();
    const m = new THREE.Mesh(new RoundedBoxGeometry(0.03, h, w, 2, Math.min(h, w) * 0.45), portMat);
    g.add(m);
    const ring = new THREE.Mesh(new RoundedBoxGeometry(0.031, h + 0.03, w + 0.03, 2, (Math.min(h, w) + 0.03) * 0.45), ringMat());
    ring.material.color.set(is14 ? '#5aa2ff' : '#ff5a70'); ring.renderOrder = 3;
    g.add(ring);
    g.position.set(side * (Wd / 2 - 0.006), Tb * 0.48, z);
    body.add(g);
    ports[name] = { group: g, ring, side };
    return g;
  }
  if (is14) {
    port('magsafe', -1, -0.72, 0.1, 0.022);
    port('tb1', -1, -0.5, 0.085, 0.028); port('tb2', -1, -0.33, 0.085, 0.028);
    port('jack', -1, 0.62, 0.035, 0.035);
    port('hdmi', 1, -0.62, 0.14, 0.038); port('tb3', 1, -0.4, 0.085, 0.028); port('sd', 1, 0.35, 0.24, 0.014);
  } else {
    port('tb1', -1, -0.62, 0.085, 0.028); port('tb2', -1, -0.4, 0.085, 0.028);
    port('jack', 1, -0.55, 0.035, 0.035);
  }

  // lid: pivots at the hinge; its inner face carries the bezel glass and the display
  const lidPivot = new THREE.Group();
  lidPivot.position.set(0, Tb + 0.003, -D / 2 + 0.02);
  body.add(lidPivot);
  const lidShell = new THREE.Mesh(slab(Wd, D - 0.02, Tl, is14 ? 0.12 : 0.14, is14 ? 0.016 : 0.028), alu);
  lidShell.position.set(0, 0, (D - 0.02) / 2);
  lidPivot.add(lidShell);
  const glass = new THREE.Mesh(flatRR(Wd - 0.03, D - 0.05, is14 ? 0.1 : 0.12), Mat({ color: '#030304', roughness: 0.25, metalness: 0, envMapIntensity: 0.25, clearcoat: 1, clearcoatRoughness: 0.1 }));
  glass.rotation.x = Math.PI; glass.position.set(0, -0.001, (D - 0.02) / 2);
  lidPivot.add(glass);
  // display area (bezels: 13" thick, 14" thin)
  const dW = is14 ? 3.025 : 2.865, dH = is14 ? 1.965 : 1.79;
  const chin = is14 ? 0.2 : 0.2;
  const scr = canvasTexOf(is14 ? 1232 : 1280, 800);
  const screenMat = new THREE.MeshBasicMaterial({ map: scr.tex, toneMapped: false });
  const screenGeo = new THREE.PlaneGeometry(dW, dH);
  screenGeo.rotateX(Math.PI / 2);   // faces -y; texture top toward +z (the lid's free edge)
  const screen = new THREE.Mesh(screenGeo, screenMat);
  screen.position.set(0, -0.0022, chin + dH / 2);
  lidPivot.add(screen);
  const anchors = { screen: new THREE.Object3D(), screenTop: new THREE.Object3D(), notch: new THREE.Object3D(), bezelTop: new THREE.Object3D(), bezelSide: new THREE.Object3D(), lidTop: new THREE.Object3D(), touchBar: tbAnchor, keyboard: new THREE.Object3D(), deckL: new THREE.Object3D(), deckR: new THREE.Object3D() };
  anchors.screen.position.set(0, -0.003, chin + dH / 2);
  anchors.screenTop.position.set(0, -0.003, chin + dH);
  anchors.notch.position.set(0, -0.003, chin + dH - 0.03);
  anchors.bezelTop.position.set(0, -0.003, (chin + dH + D - 0.02) / 2);
  anchors.bezelSide.position.set(-(Wd / 2 + dW / 2) / 2, -0.003, chin + dH / 2);
  anchors.lidTop.position.set(0, Tl, D - 0.02);
  [anchors.screen, anchors.screenTop, anchors.notch, anchors.bezelTop, anchors.bezelSide, anchors.lidTop].forEach(a => lidPivot.add(a));
  anchors.keyboard.position.set(0, Tb, kz0 + 2.5 * u); body.add(anchors.keyboard);
  anchors.deckL.position.set(-gx, Tb, kz0 + 2.5 * u); body.add(anchors.deckL);
  anchors.deckR.position.set(gx, Tb, kz0 + 2.5 * u); body.add(anchors.deckR);
  if (!is14) { // 13": camera dot in the top bezel
    const cam = new THREE.Mesh(new THREE.CircleGeometry(0.012, 16), new THREE.MeshBasicMaterial({ color: '#15182a' }));
    cam.rotation.x = Math.PI / 2; cam.position.set(0, -0.003, D - 0.02 - 0.07); lidPivot.add(cam);
  }

  const st = { lid: 105, lift: 0, glow: 0.85, shadow: 1 };
  const L = {
    kind, group, body, lidPivot, screen, screenMat, touchBar, ports, anchors, st, grilles, W: Wd, D, Tb,
    canvas: scr.canvas, ctx: scr.ctx, tex: scr.tex, draw: null,
    update() {
      lidPivot.rotation.x = -st.lid * Math.PI / 180;
      screenMat.color.setScalar(Math.min(1, st.glow));
      shadow.material.opacity = 0.85 * st.shadow * Math.max(0, 1 - group.position.y * 1.2);
      shadow.position.y = 0.004 - group.position.y;
      shadow.visible = shadow.material.opacity > 0.01;
    },
  };
  return L;
}
function canvasTexOf(w, h) { return canvasTex(w, h, true); }

window.STAGE_CREATE = createStage;
