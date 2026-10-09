import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { createLoungeChair } from './loungeChair';
import { createFurniture, rbox, at } from './furniture';
import { wallX, wallZ, slab, rectPanel, archPanel, type Opening } from './arch';
import { jaaliAlpha } from './surfaces';
import type { Kit } from './kit';
import { houseAnnotations, houseLayers, type HouseLayerId } from '../house';

/**
 * The Pool Pavilion Villa, built as the construction sequence a client
 * would see on site. Each layer is its own group so the explorer can
 * assemble it step by step, pull it apart into an exploded diagram, or
 * cut a section through it. Metres; front (south) faces +Z, east is +X.
 */

export const LAYERS = houseLayers;
export type LayerId = HouseLayerId;

export interface Annotation {
  n: number;
  title: string;
  text: string;
  anchor: THREE.Vector3;
}

export const ANNOTATIONS: Annotation[] = houseAnnotations.map((a) => ({ n: a.n, title: a.title, text: a.text, anchor: new THREE.Vector3(...a.anchor) }));

const GF = 3.4; // ground floor height (top of GF slab structure)
const FF0 = 3.62; // first-floor finished level
const FFH = 3.0;

/** Merge a layer's meshes by material to keep draw calls low. */
function bake(group: THREE.Group) {
  group.updateMatrixWorld(true);
  const buckets = new Map<THREE.Material, { geos: THREE.BufferGeometry[]; cast: boolean }>();
  const keep: THREE.Object3D[] = [];
  group.traverse((o) => {
    const inst = o as THREE.InstancedMesh;
    if (inst.isInstancedMesh) {
      keep.push(inst);
      return;
    }
    const m = o as THREE.Mesh;
    if (!m.isMesh || Array.isArray(m.material)) return;
    let g = m.geometry.clone();
    g.applyMatrix4(m.matrixWorld);
    if (g.index) g = g.toNonIndexed();
    for (const name of Object.keys(g.attributes)) if (!['position', 'normal', 'uv'].includes(name)) g.deleteAttribute(name);
    if (!g.getAttribute('normal')) g.computeVertexNormals();
    if (!g.getAttribute('uv')) g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(g.getAttribute('position').count * 2), 2));
    g.morphAttributes = {};
    const b = buckets.get(m.material) ?? { geos: [], cast: false };
    b.geos.push(g);
    b.cast ||= m.castShadow;
    buckets.set(m.material, b);
  });
  const baked = new THREE.Group();
  baked.name = group.name;
  for (const [mat, { geos, cast }] of buckets) {
    const merged = mergeGeometries(geos, false);
    geos.forEach((g) => g.dispose());
    if (!merged) continue;
    const mesh = new THREE.Mesh(merged, mat);
    mesh.castShadow = cast;
    mesh.receiveShadow = true;
    baked.add(mesh);
  }
  for (const k of keep) {
    k.updateMatrixWorld(true);
    k.matrixWorld.decompose(k.position, k.quaternion, k.scale);
    k.castShadow = true;
    k.receiveShadow = true;
    baked.add(k);
  }
  // Free the source geometries (instanced meshes are re-used)
  group.traverse((o) => {
    const m = o as THREE.Mesh;
    if (m.isMesh && !(o as THREE.InstancedMesh).isInstancedMesh) m.geometry.dispose();
  });
  return baked;
}

function shadowAll(o: THREE.Object3D) {
  o.traverse((c) => {
    if ((c as THREE.Mesh).isMesh) {
      c.castShadow = true;
      c.receiveShadow = true;
    }
  });
  return o;
}

export async function buildVilla(M: Kit, onProgress?: (p: number) => void) {
  const step = async (k: number) => {
    onProgress?.(k / 9);
    await new Promise((r) => setTimeout(r, 0));
  };
  const F = createFurniture(M);
  const concrete = M.concrete();
  const stone = M.sandstone(6);
  const plaster = M.plaster([240, 234, 222], 'villa');
  const plasterIn = M.plaster([236, 228, 214], 'in');
  const teak = M.wood('teak', 73);
  const deck = M.sandstone(0, [222, 204, 172]);

  const groups = Object.fromEntries(LAYERS.map((l) => [l.id, new THREE.Group()])) as Record<LayerId, THREE.Group>;

  await step(0);
  /* ── 1. Foundation ─────────────────────────────────────────────── */
  {
    const g = groups.foundation;
    g.add(slab(-7.25, 7.25, -0.45, 0, -9.25, 0.2, concrete));
    for (const x of [-6.85, -2.3, 2.3, 6.85])
      for (const z of [-8.85, -4.6, -0.25]) g.add(slab(x - 0.6, x + 0.6, -1.0, -0.45, z - 0.6, z + 0.6, concrete));
    // Verandah steps down to the deck
    g.add(slab(-1.2, 1.2, -0.45, -0.15, 0.2, 0.55, deck));
  }

  await step(1);
  /* ── 2. Structure: columns, beams, slabs, stair ────────────────── */
  {
    const g = groups.structure;
    const col = (x: number, z: number, y0: number, y1: number) => g.add(slab(x - 0.13, x + 0.13, y0, y1, z - 0.13, z + 0.13, concrete));
    for (const x of [-6.85, -2.3, 2.3, 6.85]) for (const z of [-8.85, -4.6, -0.25]) col(x, z, 0, GF);
    for (const x of [-7.85, -2.75, 2.35]) for (const z of [-7.05, -3.0, 1.1]) col(x, z, FF0, FF0 + FFH);
    // Ground-floor roof (first-floor) slab with a stair void
    const sy0 = GF;
    const sy1 = FF0;
    g.add(slab(-7.1, 4.45, sy0, sy1, -9.0, 0.15, concrete));
    g.add(slab(5.85, 7.1, sy0, sy1, -9.0, 0.15, concrete));
    g.add(slab(4.45, 5.85, sy0, sy1, -4.9, 0.15, concrete));
    // Cantilevered first-floor slab beyond the ground-floor footprint
    g.add(slab(-8.0, 2.5, sy0, sy1, 0.15, 1.25, concrete));
    g.add(slab(-8.0, -7.1, sy0, sy1, -7.2, 0.15, concrete));
    // Edge beams
    g.add(slab(-7.1, 7.1, GF - 0.35, GF, -0.05, 0.15, concrete));
    g.add(slab(-7.1, 7.1, GF - 0.35, GF, -9.0, -8.8, concrete));
    // Upper roof slab with a generous overhang
    g.add(slab(-8.5, 3.0, FF0 + FFH, FF0 + FFH + 0.22, -7.6, 1.85, concrete));
    // Stair: a straight flight in the dining wing up to the terrace
    for (let i = 0; i < 14; i++) {
      const y = ((i + 1) * FF0) / 14;
      g.add(slab(4.55, 5.75, y - 0.18, y, -8.6 + i * 0.27, -8.6 + (i + 1) * 0.27, concrete));
    }
  }

  await step(2);
  /* ── 3. Walls ──────────────────────────────────────────────────── */
  {
    const g = groups.walls;
    // Front arcade of five arches
    const arches: Opening[] = [-5.6, -3.2, -0.8, 1.6, 4.0].map((x) => ({ x: x + 7, w: 1.8, y: 0, h: 2.9, arch: true }));
    g.add(wallX(-7, -0.45, 14, GF, 0.45, arches, stone));
    // West wall with two arched jaali windows into the living room
    g.add(wallZ(-7, 0, 9, GF, 0.3, [{ x: 1.0, w: 1.2, y: 0, h: GF - 0.25 }, { x: 4.9, w: 1.0, y: 0.45, h: 2.4, arch: true }, { x: 6.9, w: 1.0, y: 0.45, h: 2.4, arch: true }], stone));
    // East wall with kitchen windows
    g.add(wallZ(6.7, 0, 9, GF, 0.3, [{ x: 1.0, w: 1.2, y: 0, h: GF - 0.25 }, { x: 5.4, w: 1.4, y: 1.0, h: 1.3 }], stone));
    // Back (north) wall with arched windows
    g.add(wallX(-7, -9, 14, GF, 0.3, [-5.6, -2.1, 1.0].map((x) => ({ x: x + 7, w: 1.2, y: 0.7, h: 2.1, arch: true })), stone));
    // Partition between living and dining, with an arched doorway
    g.add(wallZ(-0.12, -3.05, 5.65, GF, 0.24, [{ x: 2.2, w: 1.6, y: 0, h: 2.7, arch: true }], plasterIn));

    // First floor
    const ff = (m: THREE.Mesh) => {
      m.position.y = FF0;
      g.add(m);
    };
    ff(wallX(-8, 1.0, 10.5, FFH, 0.25, [{ x: 0.7, w: 4.6, y: 0.55, h: 1.75 }, { x: 6.4, w: 3.7, y: 0, h: FFH - 0.25 }], plaster));
    ff(wallZ(-8, 1.25, 8.45, FFH, 0.25, [{ x: 3.4, w: 1.4, y: 0.6, h: 1.8 }], plaster));
    ff(wallZ(2.25, 1.0, 8.2, FFH, 0.25, [{ x: 1.6, w: 1.2, y: 0, h: 2.4 }], plaster));
    ff(wallX(-8, -7.2, 10.5, FFH, 0.25, [{ x: 1.4, w: 1.4, y: 0.8, h: 1.5 }, { x: 6.8, w: 1.4, y: 0.8, h: 1.5 }], plaster));
    ff(wallZ(-2.88, 1.0, 7.95, FFH, 0.2, [{ x: 4.2, w: 1.0, y: 0, h: 2.3 }], plasterIn));
    // Terrace parapets
    g.add(slab(2.5, 7.1, FF0, FF0 + 0.95, 0.0, 0.15, plaster));
    g.add(slab(6.95, 7.1, FF0, FF0 + 0.95, -9.0, 0.15, plaster));
    g.add(slab(2.5, 7.1, FF0, FF0 + 0.95, -9.0, -8.85, plaster));
    g.add(slab(2.25, 2.5, FF0, FF0 + 0.95, -9.0, -7.2, plaster));
    // Lime-washed roof finish, parapet coping and slab fascias
    const roofTop = FF0 + FFH + 0.22;
    g.add(slab(-8.5, 3.0, roofTop, roofTop + 0.04, -7.6, 1.85, plaster));
    g.add(slab(-8.52, 3.02, roofTop - 0.24, roofTop + 0.06, 1.85, 1.87, plaster));
    g.add(slab(-7.12, 7.12, GF - 0.35, FF0, 0.15, 0.17, plaster));
    g.add(slab(-8.02, 2.5, GF, FF0, 1.25, 1.27, plaster));
  }

  await step(3);
  /* ── 4. Screens & glazing ─────────────────────────────────────── */
  {
    const g = groups.screens;
    const glass = M.glass();
    // Recessed ground-floor glazing behind the arcade, teak mullions
    g.add(slab(-6.7, -0.15, 0, GF - 0.35, -3.08, -3.02, glass));
    g.add(slab(0.15, 6.7, 0, GF - 0.35, -3.08, -3.02, glass));
    for (let x = -6.5; x <= 6.6; x += 1.3) g.add(slab(x - 0.04, x + 0.04, 0, GF - 0.35, -3.02, -2.96, teak));
    g.add(slab(-6.7, 6.7, GF - 0.42, GF - 0.35, -3.02, -2.96, teak));
    // West jaali windows (sandstone lattice)
    const jaali = M.jaaliStone([236, 228, 212], 0.5);
    for (const z of [-5.4, -7.4]) {
      const p = archPanel(1.0, 2.4, jaali);
      p.rotation.y = Math.PI / 2;
      p.position.set(-6.85, 0.45, z);
      g.add(p);
    }
    // Back windows and side doors
    for (const x of [-5.0, -1.5, 1.6]) g.add(slab(x - 0.6, x + 0.6, 0.7, 2.3, -8.88, -8.84, glass));
    // Upper floor ribbon window with teak fins, study glazing behind the jaali
    g.add(slab(-7.3, -2.7, FF0 + 0.55, FF0 + 2.3, 1.05, 1.1, glass));
    for (let x = -7.0; x <= -2.9; x += 0.68) g.add(slab(x - 0.025, x + 0.025, FF0 + 0.55, FF0 + 2.3, 1.12, 1.34, teak));
    g.add(slab(-1.6, 2.1, FF0, FF0 + FFH - 0.25, 0.7, 0.75, glass));
    const bj = rectPanel(3.9, FFH - 0.05, M.brickJaali());
    bj.position.set(0.25, FF0, 1.32);
    g.add(bj);
    // Pergola on the terrace
    for (const x of [3.1, 6.8]) for (const z of [-0.1, -4.5]) g.add(slab(x - 0.08, x + 0.08, FF0, FF0 + 2.6, z - 0.08, z + 0.08, teak));
    for (let x = 3.0; x <= 6.95; x += 0.32) g.add(slab(x - 0.04, x + 0.04, FF0 + 2.6, FF0 + 2.75, -4.7, 0.1, teak));
    // Door leaves on the side entrances
    g.add(slab(-7.0, -6.95, 0, GF - 0.3, -2.2, -1.0, teak));
    g.add(slab(6.75, 6.8, 0, GF - 0.3, -2.2, -1.0, teak));
  }

  await step(4);
  /* ── 5. Interiors ─────────────────────────────────────────────── */
  {
    const g = groups.interiors;
    g.add(slab(-6.7, 6.7, 0, 0.06, -8.7, -0.45, M.kota()));
    g.add(slab(-7.75, 2.25, FF0 - 0.02, FF0 + 0.03, -6.95, 1.0, M.terrazzo()));
    // Terrace floor around the stair void
    const tile = M.sandstone(0, [226, 210, 180]);
    g.add(slab(2.5, 4.45, FF0, FF0 + 0.03, -8.85, 0.0, tile));
    g.add(slab(5.85, 6.95, FF0, FF0 + 0.03, -8.85, 0.0, tile));
    g.add(slab(4.45, 5.85, FF0, FF0 + 0.03, -4.9, 0.0, tile));
    // Kitchen: teak base, terrazzo counter, open shelf
    g.add(slab(0.4, 4.2, 0.06, 0.92, -8.7, -8.1, teak));
    g.add(slab(0.35, 4.25, 0.92, 0.96, -8.72, -8.05, M.terrazzo()));
    g.add(slab(0.6, 4.0, 1.75, 1.79, -8.7, -8.4, teak));
    // Rugs
    const rug = (w: number, d: number, x: number, y: number, z: number) => {
      const r = new THREE.Mesh(new THREE.PlaneGeometry(w, d), M.rug(w, d));
      r.rotation.x = -Math.PI / 2;
      r.position.set(x, y, z);
      g.add(r);
    };
    rug(3.4, 2.4, -3.4, 0.065, -6.5);
    rug(2.6, 1.8, -5.2, FF0 + 0.035, -3.6);
    // Brass pendants over the dining table and the living room
    for (const x of [1.7, 2.4, 3.1]) g.add(at(F.pendant(1.05, 0.18, 2), x, GF, -5.6));
    g.add(at(F.pendant(1.2, 0.24, 2), -3.4, GF, -6.6));
    // Niches in the partition with ceramics
    for (const z of [-7.6, -4.2]) {
      g.add(at(F.vase('tall', M.glazeIndigo(), 1), -0.4, 0.06, z));
      g.add(at(F.terracottaPot(0.18, 0.34), 0.45, 0.06, z + 0.3));
    }
  }

  await step(5);
  /* ── 6. Furniture ─────────────────────────────────────────────── */
  {
    const g = groups.furniture;
    const chairMats = {
      wood: M.wood('sheesham', 7),
      cane: M.cane(),
      brass: M.brass(),
      cushion: M.fabric([233, 223, 205]),
      bolster: M.fabric([58, 70, 104]),
    };
    const y0 = 0.06;
    // Living
    g.add(at(F.sofa(2.4), -3.4, y0, -8.0));
    g.add(at(F.coffeeTable(0.48), -3.4, y0, -6.6));
    g.add(at(createLoungeChair(chairMats), -4.5, y0, -5.1, Math.PI - 0.35));
    g.add(at(createLoungeChair(chairMats), -2.3, y0, -5.1, Math.PI + 0.35));
    g.add(at(F.floorLamp(1.5), -1.5, y0, -8.25));
    g.add(at(F.mudha(), -5.45, y0, -7.9));
    const plant = new THREE.Group();
    plant.add(F.terracottaPot(0.24, 0.42));
    plant.add(at(F.leafyPlant(1.5, 4), 0, 0.38, 0));
    g.add(at(plant, -6.2, y0, -8.2));
    // Dining
    g.add(at(F.diningTable(2.2, 0.95), 2.4, y0, -5.6));
    for (const side of [-1, 1]) for (const dx of [-0.72, 0, 0.72]) g.add(at(F.diningChair('sheesham'), 2.4 + dx, y0, -5.6 + side * 0.7, side > 0 ? Math.PI : 0));
    // Verandah: jhoola hung from the slab, charpai, planters
    g.add(at(F.jhoola(1.25, GF - y0 - 0.55), -4.6, y0, -1.75));
    g.add(at(F.charpai(1.9, 0.9), 3.6, y0, -1.75));
    for (const x of [-6.3, 6.3]) {
      const p = new THREE.Group();
      p.add(F.terracottaPot(0.3, 0.55));
      p.add(at(F.snakePlant(0.9, 16, x > 0 ? 3 : 7), 0, 0.5, 0));
      g.add(at(p, x, y0, -1.0));
    }
    // Upper floor: bedroom, study, terrace
    const y1 = FF0 + 0.03;
    g.add(at(bed(M, F), -5.3, y1, -5.4));
    g.add(at(F.jaaliConsole(1.3, 0.38, 0.78), -7.5, y1, -2.0, Math.PI / 2));
    g.add(at(createLoungeChair(chairMats), -3.7, y1, -0.3, -2.6));
    g.add(at(F.chowki(0.82, 0.34), -0.3, y1, -2.4));
    g.add(at(F.mudha(), -1.2, y1, -2.1));
    g.add(at(F.mudha(0.38), 0.6, y1, -2.7));
    g.add(at(F.reededSideboard(1.8, 0.45, 0.78), -0.3, y1, -6.55));
    g.add(at(createLoungeChair(chairMats), 4.2, y1, -2.0, 0.5));
    g.add(at(F.chowki(0.7, 0.3), 5.4, y1, -2.4));
    shadowAll(g);
  }

  await step(6);
  /* ── 7. Landscape ─────────────────────────────────────────────── */
  {
    const g = groups.landscape;
    g.add(slab(-40, 40, -0.32, -0.06, -40, 40, M.lawn()));
    g.add(slab(-10, 10, -0.06, 0, 0.55, 2, deck));
    g.add(slab(-10, -6.2, -0.06, 0, 2, 6.6, deck));
    g.add(slab(5.2, 10, -0.06, 0, 2, 6.6, deck));
    g.add(slab(-10, 10, -0.06, 0, 6.6, 8.2, deck));
    g.add(slab(-10, -7.25, -0.06, 0, -9.5, 0.55, deck));
    g.add(slab(7.25, 10, -0.06, 0, -9.5, 0.55, deck));
    g.add(slab(-6.2, 5.2, -0.055, -0.035, 2, 6.6, M.water()));
    const coping = M.sandstone(0, [236, 226, 206]);
    for (const [x0, x1, z0, z1] of [
      [-6.5, 5.5, 1.75, 2.0],
      [-6.5, 5.5, 6.6, 6.85],
      [-6.5, -6.2, 2, 6.6],
      [5.2, 5.5, 2, 6.6],
    ])
      g.add(slab(x0, x1, 0, 0.05, z0, z1, coping));
    for (const [x, z, h, s, seed] of [
      [-9.4, 3.6, 5.6, 2.6, 4],
      [9.2, -2.5, 6.4, 3.0, 9],
      [-12.5, -6, 7.5, 3.6, 13],
      [12.5, -9, 8, 3.8, 21],
      [-3, -14, 9, 4.5, 31],
      [5, -15, 8, 4, 35],
    ])
      g.add(at(F.tree(h, s, seed), x, 0, z));
    for (let i = 0; i < 7; i++) g.add(at(F.shrub(0.5, i + 11), -6.4 + i * 2.1, 0, 0.42 + (i % 2) * 0.05));
    for (const x of [6.8, 8.2]) g.add(at(lounger(M, F), x, 0, 4.6, Math.PI + 0.1));
    shadowAll(g);
  }

  await step(7);
  const root = new THREE.Group();
  root.name = 'Villa';
  const layers = LAYERS.map((meta) => {
    const baked = bake(groups[meta.id]);
    shadowAll(baked);
    const holder = new THREE.Group();
    holder.name = meta.id;
    holder.add(baked);
    root.add(holder);
    const box = new THREE.Box3().setFromObject(holder);
    return { ...meta, group: holder, box };
  });
  onProgress?.(1);
  return { root, layers };
}

/** Low sheesham bed with an arched jaali headboard. */
function bed(M: Kit, F: ReturnType<typeof createFurniture>) {
  const g = new THREE.Group();
  const wood = M.wood('sheesham', 19);
  g.add(at(rbox(1.9, 0.26, 2.15, wood, 0.02), 0, 0.17, 0));
  g.add(at(F.cushion(1.8, 0.2, 2.0, M.fabric([233, 224, 207])), 0, 0.4, 0.05));
  for (const x of [-0.45, 0.45]) g.add(at(F.cushion(0.62, 0.14, 0.38, M.fabric(x < 0 ? [52, 62, 94] : [170, 96, 70])), x, 0.56, -0.8));
  g.add(at(rbox(1.84, 0.05, 0.75, M.rug(1.84, 0.75), 0.01), 0, 0.52, 0.62));
  const head = archPanel(1.7, 1.25, M.woodAlpha('sheesham', jaaliAlpha(512, 4), 0.25));
  head.position.set(0, 0.3, -1.06);
  g.add(head);
  g.add(at(rbox(1.9, 0.06, 0.06, wood, 0.01), 0, 0.33, -1.06));
  return shadowAll(g);
}

function lounger(M: Kit, F: ReturnType<typeof createFurniture>) {
  const g = new THREE.Group();
  const w = M.wood('teak', 71);
  for (let i = 0; i < 9; i++) g.add(at(rbox(0.62, 0.025, 0.08, w, 0.005), 0, 0.32, -0.9 + i * 0.105));
  const back = rbox(0.62, 0.025, 0.7, w, 0.005);
  back.position.set(0, 0.5, -1.1);
  back.rotation.x = 0.85;
  g.add(back);
  for (const x of [-0.28, 0.28]) for (const z of [-0.9, 0]) g.add(at(rbox(0.05, 0.3, 0.05, w, 0.005), x, 0.15, z));
  g.add(at(F.cushion(0.58, 0.06, 1.0, M.fabric([233, 224, 207])), 0, 0.36, -0.45));
  return shadowAll(g);
}
