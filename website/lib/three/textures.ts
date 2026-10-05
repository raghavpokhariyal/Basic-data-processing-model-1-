import * as THREE from 'three';
import { clamp01, fbm, mix, noise2, smoothstep } from './noise';

/**
 * Procedural canvas textures for the furniture model. Painting them at
 * runtime keeps the 3D viewer free of large texture downloads.
 */

type RGB = [number, number, number];

export type WoodSpecies = 'sheesham' | 'teak' | 'mango';

export const WOOD_PALETTES: Record<WoodSpecies, { light: RGB; base: RGB; dark: RGB; figure: number }> = {
  // Indian rosewood: deep chocolate with dark streaks
  sheesham: { light: [146, 92, 58], base: [110, 66, 40], dark: [58, 33, 20], figure: 1.15 },
  // Golden-brown, straight grained
  teak: { light: [196, 146, 92], base: [164, 114, 66], dark: [112, 74, 40], figure: 0.8 },
  // Pale and lively, with soft grey spalting
  mango: { light: [214, 184, 142], base: [186, 152, 110], dark: [128, 104, 80], figure: 1.0 },
};

function makeCanvas(w: number, h = w) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

function toTexture(canvas: HTMLCanvasElement, srgb = true, repeat = true) {
  const t = new THREE.CanvasTexture(canvas);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 8;
  t.needsUpdate = true;
  return t;
}

/** Seamless wood grain. Grain runs along the texture's U (x) axis. */
export function woodCanvas(species: WoodSpecies, size = 512, seed = 7) {
  const p = WOOD_PALETTES[species];
  const c = makeCanvas(size);
  const ctx = c.getContext('2d')!;
  const img = ctx.createImageData(size, size);
  const d = img.data;
  const rings = 18;
  for (let y = 0; y < size; y++) {
    const ny = y / size;
    for (let x = 0; x < size; x++) {
      const nx = x / size;
      // Gentle, mostly straight figure: long low-frequency warp plus a little detail
      const warp = (fbm(nx, ny * 3, 1, 3, 3, seed) - 0.5) * 1.6 * p.figure + (fbm(nx * 4, ny * 16, 4, 16, 2, seed + 1) - 0.5) * 0.35;
      const ring = ny * rings + warp;
      const t = ring - Math.floor(ring);
      // Latewood: a sharp dark line followed by a soft fade
      const late = smoothstep(0.7, 0.92, t) * (1 - smoothstep(0.92, 1.0, t));
      const pores = fbm(nx * 64, ny * 256, 64, 256, 2, seed + 3);
      const streak = fbm(nx * 2, ny * 40, 2, 40, 2, seed + 5);
      const broad = fbm(nx * 2, ny * 2, 2, 2, 3, seed + 9);
      const spalt = species === 'mango' ? (1 - smoothstep(0, 0.014, Math.abs(fbm(nx * 3, ny * 3, 3, 3, 4, seed + 21) - 0.55))) * 0.4 : 0;
      for (let k = 0; k < 3; k++) {
        let v = mix(p.base[k], p.light[k], smoothstep(0.3, 0.7, broad) * 0.7 + streak * 0.3);
        v = mix(v, p.dark[k], late * 0.6);
        v *= 0.9 + pores * 0.16;
        v = mix(v, 92, spalt);
        d[(y * size + x) * 4 + k] = v;
      }
      d[(y * size + x) * 4 + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

export function woodTexture(species: WoodSpecies, size = 512, seed = 7) {
  return toTexture(woodCanvas(species, size, seed));
}

/**
 * Hand-woven "Vienna" cane webbing: horizontal and vertical strand pairs
 * plus two diagonal families, leaving octagonal openings. Transparent
 * where open, so use it with `alphaTest`.
 */
export function caneCanvas(size = 512, cells = 4, tone: RGB = [220, 186, 122]) {
  const c = makeCanvas(size);
  const ctx = c.getContext('2d')!;
  const s = size / cells;
  const edge = `rgb(${tone.map((v) => Math.round(v * 0.62)).join(',')})`;
  const face = `rgb(${tone.join(',')})`;
  const hi = `rgb(${tone.map((v) => Math.min(255, Math.round(v * 1.1))).join(',')})`;

  const strand = (x1: number, y1: number, x2: number, y2: number, w: number) => {
    ctx.lineCap = 'butt';
    ctx.strokeStyle = edge;
    ctx.lineWidth = w;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.strokeStyle = face;
    ctx.lineWidth = w * 0.72;
    ctx.stroke();
    ctx.strokeStyle = hi;
    ctx.lineWidth = w * 0.2;
    ctx.stroke();
  };

  const pairW = s * 0.085;
  const gap = s * 0.055;
  // Horizontal and vertical pairs, centred on the cell grid
  for (let i = -1; i <= cells + 1; i++) {
    const g = i * s;
    for (const o of [-gap, gap]) {
      strand(-s, g + o, size + s, g + o, pairW);
    }
  }
  for (let i = -1; i <= cells + 1; i++) {
    const g = i * s;
    for (const o of [-gap, gap]) {
      strand(g + o, -s, g + o, size + s, pairW);
    }
  }
  // Diagonals pass through strand midpoints, cutting each opening's corners
  const diagW = s * 0.1;
  for (let k = -cells * 2; k <= cells * 2; k++) {
    const cst = s / 2 + k * s;
    strand(cst - size, -size, cst + 2 * size, 2 * size, diagW); // x - y = cst
    strand(cst + size, -size, cst - 2 * size, 2 * size, diagW); // x + y = cst
  }
  return c;
}

export function caneTexture(size = 512, cells = 4) {
  return toTexture(caneCanvas(size, cells));
}

/** Soft linen weave for cushions. */
export function fabricCanvas(color: RGB, size = 256, seed = 4) {
  const c = makeCanvas(size);
  const ctx = c.getContext('2d')!;
  const img = ctx.createImageData(size, size);
  const d = img.data;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const warp = Math.sin((x / size) * Math.PI * 2 * 96) * 0.5 + 0.5;
      const weft = Math.sin((y / size) * Math.PI * 2 * 96) * 0.5 + 0.5;
      const slub = noise2((x / size) * 64, (y / size) * 8, 64, 8, seed);
      const shade = 0.9 + (warp * weft) * 0.08 + slub * 0.06;
      for (let k = 0; k < 3; k++) d[(y * size + x) * 4 + k] = clamp01((color[k] * shade) / 255) * 255;
      d[(y * size + x) * 4 + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

export function fabricTexture(color: RGB, size = 256, seed = 4) {
  return toTexture(fabricCanvas(color, size, seed));
}

export { makeCanvas, toTexture };
export type { RGB };
