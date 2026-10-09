import * as THREE from 'three';
import { archShape, projectUV } from './geometry';

export interface Opening {
  /** left edge along the wall (m) */
  x: number;
  w: number;
  /** sill height */
  y: number;
  /** total height including any arch */
  h: number;
  arch?: boolean;
}

/**
 * A wall slab spanning x∈[0,len], y∈[0,h], z∈[0,t] with rectangular or
 * round-arched openings cut through it. UVs are in metres.
 */
export function wall(len: number, h: number, t: number, openings: Opening[], mat: THREE.Material | THREE.Material[]) {
  const s = new THREE.Shape();
  s.moveTo(0, 0);
  s.lineTo(len, 0);
  s.lineTo(len, h);
  s.lineTo(0, h);
  s.lineTo(0, 0);
  for (const o of openings) {
    const p = new THREE.Path();
    if (o.arch) {
      const r = o.w / 2;
      const spring = o.y + o.h - r;
      p.moveTo(o.x, Math.max(o.y, 0.0001));
      p.lineTo(o.x + o.w, Math.max(o.y, 0.0001));
      p.lineTo(o.x + o.w, spring);
      p.absarc(o.x + r, spring, r, 0, Math.PI, false);
      p.lineTo(o.x, Math.max(o.y, 0.0001));
    } else {
      p.moveTo(o.x, Math.max(o.y, 0.0001));
      p.lineTo(o.x + o.w, Math.max(o.y, 0.0001));
      p.lineTo(o.x + o.w, o.y + o.h);
      p.lineTo(o.x, o.y + o.h);
      p.lineTo(o.x, Math.max(o.y, 0.0001));
    }
    s.holes.push(p);
  }
  const geo = new THREE.ExtrudeGeometry(s, { depth: t, bevelEnabled: false, curveSegments: 28 });
  const m = new THREE.Mesh(geo, mat);
  m.castShadow = true;
  m.receiveShadow = true;
  return m;
}

/** Wall along world Z (rotated), spanning z from z0 down to z0-len, x from x0 to x0+t. */
export function wallZ(x0: number, z0: number, len: number, h: number, t: number, openings: Opening[], mat: THREE.Material | THREE.Material[]) {
  const m = wall(len, h, t, openings, mat);
  m.rotation.y = Math.PI / 2;
  m.position.set(x0, 0, z0);
  return m;
}

/** Wall along world X, spanning x from x0 to x0+len, z from z0 to z0+t. */
export function wallX(x0: number, z0: number, len: number, h: number, t: number, openings: Opening[], mat: THREE.Material | THREE.Material[]) {
  const m = wall(len, h, t, openings, mat);
  m.position.set(x0, 0, z0);
  return m;
}

export function slab(x0: number, x1: number, y0: number, y1: number, z0: number, z1: number, mat: THREE.Material) {
  const g = projectUV(new THREE.BoxGeometry(x1 - x0, y1 - y0, z1 - z0));
  g.translate((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
  // Re-project after translate so UVs are world-aligned (seamless across slabs)
  projectUV(g);
  const m = new THREE.Mesh(g, mat);
  m.castShadow = true;
  m.receiveShadow = true;
  return m;
}

/** Flat arched panel (e.g. a jaali screen) in the XY plane, base at y=0. */
export function archPanel(w: number, h: number, mat: THREE.Material) {
  const r = w / 2;
  const g = new THREE.ShapeGeometry(archShape(r, h - r, 32), 1);
  const m = new THREE.Mesh(g, mat);
  m.castShadow = true;
  m.receiveShadow = true;
  return m;
}

export function rectPanel(w: number, h: number, mat: THREE.Material) {
  const g = new THREE.PlaneGeometry(w, h);
  const uv = g.getAttribute('uv');
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * w, uv.getY(i) * h);
  g.translate(0, h / 2, 0);
  const m = new THREE.Mesh(g, mat);
  m.castShadow = true;
  m.receiveShadow = true;
  return m;
}

/** Soft gradient backdrop (unlit) for views out of windows. */
export function backdrop(w: number, h: number, stops: [number, string][]) {
  const c = document.createElement('canvas');
  c.width = 16;
  c.height = 256;
  const ctx = c.getContext('2d')!;
  const gr = ctx.createLinearGradient(0, 0, 0, 256);
  for (const [o, col] of stops) gr.addColorStop(o, col);
  ctx.fillStyle = gr;
  ctx.fillRect(0, 0, 16, 256);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: t }));
  return m;
}
