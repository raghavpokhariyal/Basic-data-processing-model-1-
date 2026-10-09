/**
 * Procedural surface textures for the 3D house and furniture: lime
 * plaster, sandstone, kota stone, terrazzo, brick, jaali lattices,
 * block-printed rugs, niwar and reed weaves. Painted on canvases at
 * runtime so the viewers need no texture downloads.
 */
import * as THREE from 'three';
import { fbm, noise2, rng, smoothstep, mix } from './noise';
import { makeCanvas, toTexture, type RGB } from './textures';

type PixelFn = (nx: number, ny: number, out: number[]) => void;

function paint(size: number, fn: PixelFn, h = size) {
  const c = makeCanvas(size, h);
  const ctx = c.getContext('2d')!;
  const img = ctx.createImageData(size, h);
  const d = img.data;
  const out = [0, 0, 0, 255];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < size; x++) {
      out[3] = 255;
      fn(x / size, y / h, out);
      const i = (y * size + x) * 4;
      d[i] = out[0];
      d[i + 1] = out[1];
      d[i + 2] = out[2];
      d[i + 3] = out[3];
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

export function tex(canvas: HTMLCanvasElement, repeatMetres: number | [number, number] = 1, srgb = true) {
  const t = toTexture(canvas, srgb, true);
  const [rx, ry] = Array.isArray(repeatMetres) ? repeatMetres : [repeatMetres, repeatMetres];
  t.repeat.set(1 / rx, 1 / ry);
  return t;
}

const lerpRGB = (a: RGB, b: RGB, t: number, out: number[]) => {
  out[0] = mix(a[0], b[0], t);
  out[1] = mix(a[1], b[1], t);
  out[2] = mix(a[2], b[2], t);
};

/** Lime plaster / limewash: soft mottling with faint trowel marks. */
export function plaster(base: RGB, size = 512, seed = 1, strength = 1) {
  return paint(size, (x, y, o) => {
    const m = fbm(x * 4, y * 4, 4, 4, 5, seed);
    const t = fbm(x * 2, y * 10, 2, 10, 3, seed + 5);
    const k = 1 + ((m - 0.5) * 0.12 + (t - 0.5) * 0.05) * strength;
    o[0] = base[0] * k;
    o[1] = base[1] * k;
    o[2] = base[2] * k;
  });
}

/** Raw cement plaster for the "before" shots. */
export function cement(size = 512, seed = 3) {
  return paint(size, (x, y, o) => {
    const m = fbm(x * 6, y * 6, 6, 6, 5, seed);
    const sp = noise2(x * 180, y * 180, 180, 180, seed + 2);
    const stain = smoothstep(0.55, 0.75, fbm(x * 2, y * 3, 2, 3, 4, seed + 7));
    let v = 150 + (m - 0.5) * 50 + (sp - 0.5) * 18 - stain * 22;
    o[0] = v;
    o[1] = v * 0.99;
    o[2] = v * 0.96;
  });
}

export function terrazzo(base: RGB, size = 1024, seed = 11) {
  const c = plaster(base, size, seed, 0.5);
  const ctx = c.getContext('2d')!;
  const r = rng(seed);
  const chips: RGB[] = [
    [200, 150, 126],
    [214, 198, 172],
    [168, 166, 160],
    [196, 170, 128],
    [246, 242, 234],
    [186, 178, 166],
  ];
  for (let i = 0; i < 6000; i++) {
    const x = r() * size;
    const y = r() * size;
    const s = (r() ** 3) * 7 + 1.2;
    const col = chips[Math.floor(r() * chips.length)];
    ctx.fillStyle = `rgb(${col.join(',')})`;
    ctx.beginPath();
    const n = 5 + Math.floor(r() * 3);
    for (let k = 0; k < n; k++) {
      const a = (k / n) * Math.PI * 2 + r() * 0.5;
      const rr = s * (0.6 + r() * 0.5);
      const px = x + Math.cos(a) * rr;
      const py = y + Math.sin(a) * rr;
      if (k === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fill();
  }
  return c;
}

/** Kota stone: grey-green limestone slabs, a classic Indian floor. */
export function kota(size = 1024, slabs = 2, seed = 21) {
  const c = paint(size, (x, y, o) => {
    const sx = Math.floor(x * slabs);
    const sy = Math.floor(y * slabs);
    const tone = noise2(sx + 0.5, sy + 0.5, 64, 64, seed) - 0.5;
    const m = fbm(x * 8, y * 8, 8, 8, 5, seed + 1);
    const vein = smoothstep(0.48, 0.5, fbm(x * 3, y * 3, 3, 3, 5, seed + 4)) * (1 - smoothstep(0.5, 0.52, fbm(x * 3, y * 3, 3, 3, 5, seed + 4)));
    const base: RGB = [128 + tone * 20, 138 + tone * 18, 126 + tone * 16];
    const k = 0.92 + m * 0.16;
    o[0] = base[0] * k - vein * 14;
    o[1] = base[1] * k - vein * 12;
    o[2] = base[2] * k - vein * 12;
    const gx = (x * slabs) % 1;
    const gy = (y * slabs) % 1;
    if (gx < 0.004 || gy < 0.004 || gx > 0.996 || gy > 0.996) {
      o[0] *= 0.7;
      o[1] *= 0.7;
      o[2] *= 0.7;
    }
  });
  return c;
}

/** Jaisalmer / Dholpur sandstone with soft bedding planes. */
export function sandstone(base: RGB = [214, 188, 148], size = 512, seed = 31, courses = 0) {
  return paint(size, (x, y, o) => {
    const bed = fbm(x * 2, y * 24, 2, 24, 3, seed);
    const m = fbm(x * 8, y * 8, 8, 8, 4, seed + 3);
    const sp = noise2(x * 200, y * 200, 200, 200, seed + 9);
    const k = 0.9 + bed * 0.12 + m * 0.08 + (sp - 0.5) * 0.06;
    o[0] = base[0] * k;
    o[1] = base[1] * k;
    o[2] = base[2] * k;
    if (courses > 0) {
      const row = Math.floor(y * courses);
      const off = row % 2 ? 0.5 / 2 : 0;
      const gy = (y * courses) % 1;
      const gx = ((x + off) * 2) % 1;
      if (gy < 0.012 || gx < 0.006) {
        o[0] *= 0.82;
        o[1] *= 0.82;
        o[2] *= 0.82;
      }
    }
  });
}

/** Exposed terracotta brick with recessed mortar joints. */
export function brick(size = 512, rows = 8, seed = 41) {
  return paint(size, (x, y, o) => {
    const row = Math.floor(y * rows);
    const off = row % 2 ? 0.125 : 0;
    const bx = Math.floor((x + off) * 4);
    const id = noise2(bx + 0.5, row + 0.5, 64, 64, seed);
    const gx = ((x + off) * 4) % 1;
    const gy = (y * rows) % 1;
    const m = fbm(x * 16, y * 16, 16, 16, 3, seed + 1);
    const mortar = gx < 0.025 || gy < 0.09;
    if (mortar) {
      const v = 196 + m * 20;
      o[0] = v;
      o[1] = v * 0.93;
      o[2] = v * 0.84;
    } else {
      const base: RGB = [166 + id * 30, 86 + id * 18, 60 + id * 10];
      const k = 0.88 + m * 0.2;
      o[0] = base[0] * k;
      o[1] = base[1] * k;
      o[2] = base[2] * k;
    }
  });
}

export function lawn(size = 512, seed = 51) {
  return paint(size, (x, y, o) => {
    const m = fbm(x * 6, y * 6, 6, 6, 5, seed);
    const g = noise2(x * 260, y * 260, 260, 260, seed + 3);
    lerpRGB([104, 118, 78], [138, 148, 98], m * 0.7 + g * 0.3, o);
  });
}

export function earth(size = 512, seed = 61) {
  return paint(size, (x, y, o) => {
    const m = fbm(x * 6, y * 6, 6, 6, 5, seed);
    const g = noise2(x * 220, y * 220, 220, 220, seed + 3);
    lerpRGB([158, 140, 116], [196, 180, 154], m * 0.7 + g * 0.3, o);
  });
}

/** Block-printed cotton dhurrie: borders and a field of hand-stamped butis. */
export function rug(size = 1024, h = 1024 * 0.68, palette: { ground: RGB; a: RGB; b: RGB } = { ground: [236, 226, 208], a: [62, 74, 108], b: [181, 101, 74] }) {
  const H = Math.round(h);
  const c = makeCanvas(size, H);
  const ctx = c.getContext('2d')!;
  const r = rng(7);
  ctx.fillStyle = `rgb(${palette.ground.join(',')})`;
  ctx.fillRect(0, 0, size, H);
  const col = (k: RGB, a = 1) => `rgba(${k.join(',')},${a})`;
  // Borders
  const bw = size * 0.05;
  ctx.fillStyle = col(palette.a);
  ctx.fillRect(0, 0, size, bw);
  ctx.fillRect(0, H - bw, size, bw);
  ctx.fillRect(0, 0, bw, H);
  ctx.fillRect(size - bw, 0, bw, H);
  ctx.strokeStyle = col(palette.b);
  ctx.lineWidth = size * 0.008;
  ctx.strokeRect(bw * 1.4, bw * 1.4, size - bw * 2.8, H - bw * 2.8);
  // Border motif: small ivory diamonds
  ctx.fillStyle = col(palette.ground, 0.9);
  for (let x = bw; x < size - bw; x += bw * 0.9) {
    for (const y of [bw / 2, H - bw / 2]) {
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(Math.PI / 4);
      ctx.fillRect(-bw * 0.16, -bw * 0.16, bw * 0.32, bw * 0.32);
      ctx.restore();
    }
  }
  // Field of butis with slight misregistration, like hand block printing
  const step = size * 0.075;
  let row = 0;
  for (let y = bw * 2.4; y < H - bw * 2; y += step * 0.86, row++) {
    for (let x = bw * 2.4 + (row % 2 ? step / 2 : 0); x < size - bw * 2; x += step) {
      const k = row % 2 ? palette.b : palette.a;
      const jx = (r() - 0.5) * 2;
      const jy = (r() - 0.5) * 2;
      ctx.save();
      ctx.translate(x + jx, y + jy);
      ctx.fillStyle = col(k, 0.85 + r() * 0.15);
      for (let p = 0; p < 6; p++) {
        ctx.rotate(Math.PI / 3);
        ctx.beginPath();
        ctx.ellipse(0, -step * 0.14, step * 0.055, step * 0.12, 0, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = col(palette.ground);
      ctx.beginPath();
      ctx.arc(0, 0, step * 0.04, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }
  // Cotton texture
  const img = ctx.getImageData(0, 0, size, H);
  const d = img.data;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      const n = 0.92 + noise2(x * 0.7, y * 0.7, 4096, 4096, 3) * 0.08 + (Math.sin(x * 1.3) * Math.sin(y * 1.3)) * 0.02;
      d[i] *= n;
      d[i + 1] *= n;
      d[i + 2] *= n;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

/** Niwar tape weave for the charpai bed. */
export function niwar(size = 512, tapes = 16) {
  return paint(size, (x, y, o) => {
    const cx = Math.floor(x * tapes);
    const cy = Math.floor(y * tapes);
    const over = (cx + cy) % 2 === 0;
    const fx = (x * tapes) % 1;
    const fy = (y * tapes) % 1;
    const along = over ? fy : fx;
    const across = over ? fx : fy;
    const stripe = over ? cx % 4 === 0 : cy % 4 === 0;
    const base: RGB = stripe ? [70, 82, 116] : [234, 224, 204];
    const shade = 0.78 + Math.sin(across * Math.PI) * 0.22 - Math.abs(along - 0.5) * 0.12;
    const thread = 0.96 + Math.sin(along * Math.PI * 30) * 0.04;
    o[0] = base[0] * shade * thread;
    o[1] = base[1] * shade * thread;
    o[2] = base[2] * shade * thread;
  });
}

/** Sarkanda reed wrapping on a mudha stool. */
export function reed(size = 512) {
  return paint(size, (x, y, o) => {
    const a = (x + y) * 18;
    const b = (x - y) * 18;
    const fa = a - Math.floor(a);
    const fb = b - Math.floor(b);
    const top = (Math.floor(a) + Math.floor(b)) % 2 === 0;
    const f = top ? fa : fb;
    const shade = 0.72 + Math.sin(f * Math.PI) * 0.28;
    const n = 0.94 + noise2(x * 90, y * 90, 90, 90, 5) * 0.08;
    o[0] = 212 * shade * n;
    o[1] = 180 * shade * n;
    o[2] = 128 * shade * n;
  });
}

/**
 * Mughal jaali: eight-point stars and small diamonds cut from a solid
 * panel. White = solid, black = open. Use as an alpha map with alphaTest.
 */
export function jaaliAlpha(size = 512, cells = 4) {
  const c = makeCanvas(size);
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, size, size);
  const s = size / cells;
  ctx.fillStyle = '#000';
  const star = (cx: number, cy: number, R: number, r: number) => {
    ctx.beginPath();
    for (let i = 0; i < 16; i++) {
      const a = (i / 16) * Math.PI * 2 + Math.PI / 16 * 0;
      const rr = i % 2 === 0 ? R : r;
      const px = cx + Math.cos(a) * rr;
      const py = cy + Math.sin(a) * rr;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fill();
  };
  for (let i = -1; i <= cells; i++) {
    for (let j = -1; j <= cells; j++) {
      star((i + 0.5) * s, (j + 0.5) * s, s * 0.42, s * 0.3);
      // Cross-shaped openings at the corners
      const cx = i * s;
      const cy = j * s;
      ctx.beginPath();
      ctx.moveTo(cx, cy - s * 0.16);
      ctx.lineTo(cx + s * 0.06, cy - s * 0.06);
      ctx.lineTo(cx + s * 0.16, cy);
      ctx.lineTo(cx + s * 0.06, cy + s * 0.06);
      ctx.lineTo(cx, cy + s * 0.16);
      ctx.lineTo(cx - s * 0.06, cy + s * 0.06);
      ctx.lineTo(cx - s * 0.16, cy);
      ctx.lineTo(cx - s * 0.06, cy - s * 0.06);
      ctx.closePath();
      ctx.fill();
    }
  }
  return c;
}

/** Hollow brick jaali (Laurie Baker style): every other brick left out. */
export function brickJaaliAlpha(size = 512, rows = 8) {
  return paint(size, (x, y, o) => {
    const row = Math.floor(y * rows);
    const off = row % 2 ? 0.125 : 0;
    const bx = Math.floor((x + off) * 4);
    const gx = ((x + off) * 4) % 1;
    const open = bx % 2 === 0 && gx > 0.12 && gx < 0.88 && (y * rows) % 1 > 0.08;
    const v = open && row % 2 === 0 ? 0 : open ? 0 : 255;
    o[0] = o[1] = o[2] = v;
  });
}

export { paint };
