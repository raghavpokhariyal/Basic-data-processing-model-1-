import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { archShape, projectUV, scaleUV, shadowed, turnedLeg } from './geometry';
import type { WoodSpecies } from './textures';
import type { Kit } from './kit';
import * as T from './surfaces';

/**
 * Procedural Indian furniture and props: charpai, chowki, jaali console,
 * reeded sideboard, jhoola, mudha, dining chair, sofa, lamps, plants and
 * trees. Dimensions in metres; pieces face +Z with feet at y = 0.
 * Parts carry `userData.label` for the exploded-view diagrams.
 */

/** Names a part for exploded diagrams. */
export function label<T extends THREE.Object3D>(o: T, text: string): T {
  o.userData.label = text;
  return o;
}

export const rbox = (w: number, h: number, d: number, mat: THREE.Material, r = 0.008, seg = 3) =>
  new THREE.Mesh(projectUV(new RoundedBoxGeometry(w, h, d, seg, Math.min(r, w / 2.01, h / 2.01, d / 2.01))), mat);

export const box = (w: number, h: number, d: number, mat: THREE.Material) => new THREE.Mesh(projectUV(new THREE.BoxGeometry(w, h, d)), mat);

export function at<T extends THREE.Object3D>(o: T, x: number, y: number, z: number, ry = 0): T {
  o.position.set(x, y, z);
  o.rotation.y = ry;
  return o;
}

/** Long part along Z built along X (keeps grain along length). */
export function railZ(len: number, h: number, w: number, mat: THREE.Material, r = 0.006) {
  const m = rbox(len, h, w, mat, r);
  m.rotation.y = Math.PI / 2;
  return m;
}
/** Vertical post built along X then turned upright. */
export function post(len: number, w: number, d: number, mat: THREE.Material, r = 0.006) {
  const g = projectUV(new RoundedBoxGeometry(len, w, d, 3, Math.min(r, w / 2.01, d / 2.01)));
  g.rotateZ(Math.PI / 2);
  g.translate(0, len / 2, 0);
  return new THREE.Mesh(g, mat);
}

export function lathe(profile: [number, number][], mat: THREE.Material, seg = 40) {
  const g = new THREE.LatheGeometry(
    profile.map(([r, y]) => new THREE.Vector2(r, y)),
    seg,
  );
  return new THREE.Mesh(g, mat);
}


export function createFurniture(M: Kit) {
  /* ---------------------------------------------------------------- objects */

  function vase(kind: 'tall' | 'round' | 'bottle' | 'bowl' = 'round', mat: THREE.Material = M.glazeIvory(), s = 1) {
    const P: Record<string, [number, number][]> = {
      tall: [[0, 0], [0.06, 0], [0.075, 0.05], [0.085, 0.16], [0.06, 0.28], [0.035, 0.36], [0.045, 0.4], [0.04, 0.4], [0.03, 0.36]],
      round: [[0, 0], [0.05, 0], [0.11, 0.06], [0.125, 0.13], [0.1, 0.21], [0.05, 0.25], [0.055, 0.27], [0.045, 0.27], [0.04, 0.25]],
      bottle: [[0, 0], [0.05, 0], [0.07, 0.04], [0.07, 0.14], [0.03, 0.22], [0.018, 0.3], [0.024, 0.31], [0.016, 0.31]],
      bowl: [[0, 0], [0.04, 0], [0.12, 0.03], [0.16, 0.08], [0.15, 0.08], [0.11, 0.035], [0.0, 0.025]],
    };
    const m = lathe(
      P[kind].map(([r, y]) => [r * s, y * s] as [number, number]),
      mat,
    );
    return m;
  }

  function terracottaPot(r = 0.22, h = 0.4) {
    return lathe(
      [
        [0, 0],
        [r * 0.62, 0],
        [r * 0.95, h * 0.55],
        [r, h * 0.88],
        [r * 1.08, h * 0.9],
        [r * 1.08, h],
        [r * 0.98, h],
        [r * 0.94, h * 0.9],
        [0, h * 0.88],
      ],
      M.terracotta(),
    );
  }

  /** Sansevieria: upright tapered blades. */
  function snakePlant(h = 0.8, n = 14, seed = 1) {
    const g = new THREE.Group();
    const rand = mulberry(seed);
    for (let i = 0; i < n; i++) {
      const hh = h * (0.6 + rand() * 0.4);
      const w = 0.045 + rand() * 0.02;
      const geo = new THREE.PlaneGeometry(w, hh, 1, 8);
      const p = geo.getAttribute('position');
      for (let k = 0; k < p.count; k++) {
        const y = p.getY(k) + hh / 2;
        const t = y / hh;
        p.setX(k, p.getX(k) * (1 - t * t * 0.9));
        p.setZ(k, Math.sin(t * 2) * 0.03);
        p.setY(k, y);
      }
      geo.computeVertexNormals();
      const leaf = new THREE.Mesh(geo, i % 3 ? M.leaf() : M.leafDark());
      leaf.rotation.y = rand() * Math.PI;
      leaf.rotation.z = (rand() - 0.5) * 0.35;
      leaf.rotation.x = (rand() - 0.5) * 0.25;
      leaf.position.set((rand() - 0.5) * 0.12, 0, (rand() - 0.5) * 0.12);
      g.add(leaf);
    }
    return g;
  }

  /** Fiddle-leaf fig style plant: a few stems with broad bent leaves. */
  function leafyPlant(h = 1.6, seed = 2, leaves = 34) {
    const g = new THREE.Group();
    const rand = mulberry(seed);
    const leafShape = new THREE.Shape();
    leafShape.moveTo(0, 0);
    leafShape.bezierCurveTo(0.09, 0.05, 0.1, 0.22, 0, 0.3);
    leafShape.bezierCurveTo(-0.1, 0.22, -0.09, 0.05, 0, 0);
    const base = new THREE.ShapeGeometry(leafShape, 6);
    const stemMat = M.bark();
    for (let s = 0; s < 3; s++) {
      const lean = new THREE.Vector3((rand() - 0.5) * 0.4, 1, (rand() - 0.5) * 0.4).normalize();
      const len = h * (0.75 + rand() * 0.25);
      const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.018, len, 8), stemMat);
      stem.position.copy(lean.clone().multiplyScalar(len / 2));
      stem.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), lean);
      g.add(stem);
      const per = Math.floor(leaves / 3);
      for (let i = 0; i < per; i++) {
        const t = 0.35 + (i / per) * 0.65;
        const p = lean.clone().multiplyScalar(len * t);
        const geo = base.clone();
        const pos = geo.getAttribute('position');
        for (let k = 0; k < pos.count; k++) {
          const x = pos.getX(k);
          const y = pos.getY(k);
          pos.setZ(k, -x * x * 2.5 + y * y * 0.6);
        }
        geo.computeVertexNormals();
        const leaf = new THREE.Mesh(geo, rand() > 0.5 ? M.leaf() : M.leafDark());
        const sc = 0.7 + rand() * 0.6 * (1 - t * 0.4);
        leaf.scale.setScalar(sc);
        leaf.position.copy(p);
        leaf.rotation.set(-0.6 - rand() * 0.6, rand() * Math.PI * 2, (rand() - 0.5) * 0.6, 'YXZ');
        g.add(leaf);
      }
    }
    return g;
  }

  /** Stylised neem / frangipani tree for exteriors. */
  function tree(h = 5, spread = 2.4, seed = 3) {
    const g = new THREE.Group();
    const rand = mulberry(seed);
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.18, h * 0.6, 10), M.bark());
    trunk.position.y = h * 0.3;
    g.add(trunk);
    for (let b = 0; b < 4; b++) {
      const dir = new THREE.Vector3(Math.cos((b / 4) * Math.PI * 2 + rand()), 1.4, Math.sin((b / 4) * Math.PI * 2 + rand())).normalize();
      const len = h * 0.35;
      const br = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.08, len, 8), M.bark());
      br.position.set(0, h * 0.55, 0).addScaledVector(dir, len / 2);
      br.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
      g.add(br);
    }
    const fol = M.foliage();
    for (let i = 0; i < 34; i++) {
      const a = rand() * Math.PI * 2;
      const r = Math.sqrt(rand()) * spread;
      const y = h * 0.7 + (rand() - 0.3) * h * 0.35 - (r / spread) * h * 0.12;
      let geo: THREE.BufferGeometry = new THREE.IcosahedronGeometry(0.5 + rand() * 0.55, 3);
      geo.deleteAttribute('normal');
      geo.deleteAttribute('uv');
      geo = mergeVertices(geo);
      const p = geo.getAttribute('position');
      for (let k = 0; k < p.count; k++) {
        const v = new THREE.Vector3().fromBufferAttribute(p, k);
        const n = Math.sin(v.x * 9 + seed) * Math.sin(v.y * 11 + i) * Math.sin(v.z * 8);
        v.multiplyScalar(0.92 + n * 0.12);
        p.setXYZ(k, v.x, v.y * 0.75, v.z);
      }
      geo.computeVertexNormals();
      const m = new THREE.Mesh(geo, fol);
      m.position.set(Math.cos(a) * r, y, Math.sin(a) * r);
      g.add(m);
    }
    return g;
  }

  function shrub(r = 0.6, seed = 5) {
    const g = new THREE.Group();
    const rand = mulberry(seed);
    for (let i = 0; i < 7; i++) {
      const m = new THREE.Mesh(new THREE.IcosahedronGeometry(r * (0.45 + rand() * 0.35), 3), M.foliage());
      m.position.set((rand() - 0.5) * r * 1.4, r * 0.45 + rand() * r * 0.3, (rand() - 0.5) * r * 1.4);
      m.scale.y = 0.8;
      g.add(m);
    }
    return g;
  }

  function palm(h = 6, seed = 9) {
    const g = new THREE.Group();
    const rand = mulberry(seed);
    const lean = new THREE.Vector3((rand() - 0.5) * 0.3, 1, (rand() - 0.5) * 0.3).normalize();
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.18, h, 10, 8), M.bark());
    trunk.position.copy(lean.clone().multiplyScalar(h / 2));
    trunk.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), lean);
    g.add(trunk);
    const top = lean.clone().multiplyScalar(h);
    for (let i = 0; i < 12; i++) {
      const geo = new THREE.PlaneGeometry(2.4, 0.5, 10, 1);
      const p = geo.getAttribute('position');
      for (let k = 0; k < p.count; k++) {
        const x = p.getX(k) + 1.2;
        const t = x / 2.4;
        p.setX(k, x);
        p.setY(k, p.getY(k) * Math.sin(t * Math.PI) * 0.9);
        p.setZ(k, -t * t * 1.2);
      }
      geo.computeVertexNormals();
      const f = new THREE.Mesh(geo, M.leaf());
      f.position.copy(top);
      f.rotation.set(0.25 + rand() * 0.3, (i / 12) * Math.PI * 2, 0, 'YXZ');
      f.rotateX(-Math.PI / 2 + 0.6);
      g.add(f);
    }
    return g;
  }

  /* -------------------------------------------------------------- furniture */

  const linen = () => M.fabric([233, 224, 207]);
  const sandLinen = () => M.fabric([206, 188, 160]);
  const indigo = () => M.fabric([52, 62, 94]);
  const rust = () => M.fabric([170, 96, 70]);

  function cushion(w: number, h: number, d: number, mat = linen()) {
    return rbox(w, h, d, mat, Math.min(h, d) * 0.45, 5);
  }

  function bolster(len: number, r: number, mat = indigo()) {
    const m = new THREE.Mesh(new THREE.CapsuleGeometry(r, len - 2 * r, 8, 20), mat);
    m.rotation.z = Math.PI / 2;
    return m;
  }

  /** Three-seat sofa on a low sheesham plinth. Faces +Z. */
  function sofa(width = 2.3, fabric = sandLinen()) {
    const g = new THREE.Group();
    const wood = M.wood('sheesham');
    const d = 0.92;
    const plinth = rbox(width, 0.12, d - 0.04, wood, 0.01);
    plinth.position.set(0, 0.11, 0);
    g.add(plinth);
    for (const x of [-width / 2 + 0.08, width / 2 - 0.08])
      for (const z of [-d / 2 + 0.1, d / 2 - 0.1]) {
        const l = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.02, 0.06, 16), M.brass());
        l.position.set(x, 0.03, z);
        g.add(l);
      }
    const base = cushion(width, 0.18, d, fabric);
    base.position.set(0, 0.26, 0);
    g.add(base);
    const back = cushion(width, 0.42, 0.2, fabric);
    back.position.set(0, 0.52, -d / 2 + 0.1);
    g.add(back);
    for (const s of [-1, 1]) {
      const arm = cushion(0.2, 0.3, d, fabric);
      arm.position.set(s * (width / 2 - 0.1), 0.46, 0);
      g.add(arm);
    }
    const n = 3;
    const sw = (width - 0.42) / n;
    for (let i = 0; i < n; i++) {
      const c = cushion(sw - 0.01, 0.14, d - 0.24, fabric);
      c.position.set(-width / 2 + 0.21 + sw * (i + 0.5), 0.41, 0.08);
      g.add(c);
      const b = cushion(sw - 0.04, 0.4, 0.16, fabric);
      b.position.set(-width / 2 + 0.21 + sw * (i + 0.5), 0.63, -d / 2 + 0.26);
      b.rotation.x = -0.18;
      g.add(b);
    }
    const t1 = cushion(0.42, 0.42, 0.12, indigo());
    t1.position.set(-width / 2 + 0.42, 0.66, -0.12);
    t1.rotation.set(-0.25, 0.3, 0.05);
    const t2 = cushion(0.4, 0.4, 0.12, rust());
    t2.position.set(width / 2 - 0.42, 0.65, -0.12);
    t2.rotation.set(-0.25, -0.35, -0.05);
    g.add(t1, t2);
    return shadowed(g);
  }

  /** Round coffee table: wooden top on a brass-ringed base. */
  function coffeeTable(r = 0.5, species: WoodSpecies = 'sheesham') {
    const g = new THREE.Group();
    const top = new THREE.Mesh(projectUV(new THREE.CylinderGeometry(r, r, 0.04, 64)), M.wood(species));
    top.position.y = 0.38;
    const base = new THREE.Mesh(projectUV(new THREE.CylinderGeometry(r * 0.55, r * 0.6, 0.34, 48)), M.wood(species, 11));
    base.position.y = 0.19;
    const ring = new THREE.Mesh(new THREE.TorusGeometry(r * 0.58, 0.008, 10, 64), M.brass());
    ring.rotation.x = Math.PI / 2;
    ring.position.y = 0.06;
    const ring2 = ring.clone();
    ring2.scale.setScalar(0.96);
    ring2.position.y = 0.32;
    g.add(top, base, ring, ring2);
    const bowl = vase('bowl', M.brass(), 1.1);
    bowl.position.set(-0.12, 0.4, 0.05);
    const v = vase('bottle', M.glazeIndigo(), 0.9);
    v.position.set(0.18, 0.4, -0.1);
    const books = bookStack(3, 2);
    books.position.set(0.12, 0.4, 0.18);
    books.rotation.y = 0.4;
    g.add(bowl, v, books);
    return shadowed(g);
  }

  function bookStack(n = 3, seed = 1) {
    const g = new THREE.Group();
    const rand = mulberry(seed);
    const cols = ['#b5654a', '#2e3a59', '#d8c3a5', '#efe9dc', '#7d6a4e'];
    let y = 0;
    for (let i = 0; i < n; i++) {
      const h = 0.025 + rand() * 0.02;
      const b = rbox(0.22 + rand() * 0.05, h, 0.16 + rand() * 0.04, M.colour(cols[Math.floor(rand() * cols.length)], 0.8), 0.003);
      b.position.y = y + h / 2;
      b.rotation.y = (rand() - 0.5) * 0.3;
      y += h;
      g.add(b);
    }
    return g;
  }

  /** Brass dome pendant with a glowing bulb. */
  function pendant(drop = 1.0, r = 0.22, emissive = 4) {
    const g = new THREE.Group();
    const shade = lathe(
      [
        [0.02, 0],
        [0.05, -0.01],
        [r * 0.6, -r * 0.35],
        [r * 0.9, -r * 0.62],
        [r, -r * 0.72],
        [r * 0.99, -r * 0.74],
        [r * 0.88, -r * 0.66],
        [r * 0.58, -r * 0.4],
        [0.04, -0.04],
      ],
      M.brass(),
      48,
    );
    shade.material = M.brass();
    (shade.material as THREE.MeshStandardMaterial).side = THREE.DoubleSide;
    shade.position.y = -drop;
    const cord = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, drop, 6), M.dark());
    cord.position.y = -drop / 2;
    const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.045, 16, 12), M.emissive('#ffd9a0', emissive));
    bulb.position.y = -drop - r * 0.5;
    g.add(shade, cord, bulb);
    shade.castShadow = true;
    return g;
  }

  function floorLamp(h = 1.55) {
    const g = new THREE.Group();
    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.17, 0.03, 40), M.brass());
    base.position.y = 0.015;
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.01, 0.01, h, 12), M.brass());
    stem.position.y = h / 2;
    const shade = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.22, 0.3, 40, 1, true), M.fabric([236, 226, 206]));
    (shade.material as THREE.MeshStandardMaterial).side = THREE.DoubleSide;
    (shade.material as THREE.MeshStandardMaterial).emissive = new THREE.Color('#ffcf8f');
    (shade.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.5;
    shade.position.y = h;
    g.add(base, stem, shade);
    return shadowed(g);
  }

  /** Framed abstract artwork: stacked arches in the brand palette. */
  function artwork(w = 0.9, h = 1.15, variant = 0) {
    const c = document.createElement('canvas');
    c.width = 600;
    c.height = Math.round((600 * h) / w);
    const ctx = c.getContext('2d')!;
    const bg = ['#e9dfcc', '#d8c3a5', '#efe7d8'][variant % 3];
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, c.width, c.height);
    const cols = ['#b5654a', '#2e3a59', '#b08d57', '#c9a27a'];
    const W = c.width;
    const H = c.height;
    for (let i = 0; i < 3; i++) {
      const aw = W * (0.62 - i * 0.16);
      const x = W / 2 - aw / 2 + (variant % 2 ? (i - 1) * W * 0.04 : 0);
      const top = H * (0.18 + i * 0.12);
      ctx.fillStyle = cols[(i + variant) % cols.length];
      ctx.beginPath();
      ctx.moveTo(x, H * 0.86);
      ctx.lineTo(x, top + aw / 2);
      ctx.arc(x + aw / 2, top + aw / 2, aw / 2, Math.PI, 0);
      ctx.lineTo(x + aw, H * 0.86);
      ctx.closePath();
      ctx.fill();
    }
    ctx.fillStyle = '#2e3a59';
    ctx.beginPath();
    ctx.arc(W * 0.72, H * 0.2, W * 0.05, 0, Math.PI * 2);
    ctx.fill();
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    const g = new THREE.Group();
    const frame = rbox(w + 0.06, h + 0.06, 0.04, M.wood('teak'), 0.006);
    const canvas = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshStandardMaterial({ map: t, roughness: 0.9 }));
    canvas.position.z = 0.021;
    g.add(frame, canvas);
    return shadowed(g);
  }

  /** Sheesham dining table with trestle legs. */
  function diningTable(len = 2.2, w = 0.95) {
    const g = new THREE.Group();
    const wood = M.wood('sheesham', 5);
    const top = rbox(len, 0.05, w, wood, 0.012);
    top.position.y = 0.75;
    g.add(top);
    for (const s of [-1, 1]) {
      const leg = rbox(0.08, 0.7, w * 0.8, wood, 0.01);
      leg.position.set(s * (len / 2 - 0.3), 0.36, 0);
      g.add(leg);
      const foot = rbox(0.12, 0.05, w * 0.84, wood, 0.01);
      foot.position.set(s * (len / 2 - 0.3), 0.025, 0);
      g.add(foot);
    }
    const str = rbox(len - 0.6, 0.06, 0.06, wood, 0.01);
    str.position.set(0, 0.3, 0);
    g.add(str);
    return shadowed(g);
  }

  /** Haveli dining chair: tapered legs, cane seat, jaali back with arched top. Faces +Z. */
  function diningChair(species: WoodSpecies = 'sheesham') {
    const g = new THREE.Group();
    const wood = M.wood(species, 13);
    const seatH = 0.46;
    const w = 0.46;
    const d = 0.48;
    const legGeo = turnedLeg(seatH, 0.016, 0.02, [{ at: 0.8, r: 0.023, w: 0.03 }]);
    for (const x of [-w / 2 + 0.025, w / 2 - 0.025]) {
      const fl = label(new THREE.Mesh(legGeo, wood), 'Turned sheesham legs');
      fl.position.set(x, 0, d / 2 - 0.03);
      g.add(fl);
      const bl = post(seatH + 0.37, 0.036, 0.03, wood, 0.008);
      bl.position.set(x, 0, -d / 2 + 0.025);
      bl.rotation.x = -0.06;
      g.add(bl);
      const sr = railZ(d - 0.04, 0.05, 0.028, wood);
      sr.position.set(x, seatH - 0.03, 0);
      g.add(sr);
    }
    for (const z of [d / 2 - 0.03, -d / 2 + 0.03]) {
      const r = rbox(w - 0.04, 0.05, 0.028, wood);
      r.position.set(0, seatH - 0.03, z);
      g.add(r);
    }
    const seat = label(new THREE.Mesh(scaleUV(new THREE.PlaneGeometry(w - 0.05, d - 0.05), w, d), M.cane()), 'Hand-woven cane seat');
    seat.rotation.x = -Math.PI / 2;
    seat.position.y = seatH - 0.008;
    g.add(seat);
    const c = label(cushion(w - 0.06, 0.05, d - 0.08, M.fabric([233, 224, 207])), 'Linen seat pad');
    c.position.set(0, seatH + 0.02, 0.01);
    g.add(c);
    // Back: arched jaali panel between the back posts
    const back = label(new THREE.Group(), 'Carved jaali back');
    const aw = (w - 0.08) / 2;
    const panel = new THREE.Mesh(
      scaleUV(new THREE.ShapeGeometry(archShape(aw, 0.2, 24), 1), 1, 1),
      M.woodAlpha(species, T.jaaliAlpha(512, 4), 0.2),
    );
    back.add(panel);
    const top = new THREE.Mesh(
      new THREE.TubeGeometry(
        new THREE.CatmullRomCurve3(
          Array.from({ length: 25 }, (_, i) => {
            const a = Math.PI - (i / 24) * Math.PI;
            return new THREE.Vector3(Math.cos(a) * aw, 0.2 + Math.sin(a) * aw, 0);
          }),
        ),
        48,
        0.014,
        10,
      ),
      wood,
    );
    back.add(top);
    const rail = rbox(w - 0.05, 0.04, 0.026, wood);
    rail.position.y = -0.01;
    back.add(rail);
    back.position.set(0, seatH + 0.17, -d / 2 + 0.03);
    back.rotation.x = -0.06;
    g.add(back);
    return shadowed(g);
  }

  /** Charpai daybed: turned legs and a niwar-woven top. Long axis X. */
  function charpai(len = 1.9, w = 0.92) {
    const g = new THREE.Group();
    const wood = M.wood('sheesham', 17);
    const h = 0.44;
    const legGeo = turnedLeg(h, 0.032, 0.03, [
      { at: 0.25, r: 0.045, w: 0.05 },
      { at: 0.55, r: 0.04, w: 0.06 },
      { at: 0.82, r: 0.046, w: 0.04 },
    ]);
    for (const x of [-len / 2, len / 2])
      for (const z of [-w / 2, w / 2]) {
        const l = label(new THREE.Mesh(legGeo, wood), 'Lathe-turned legs');
        l.position.set(x, 0, z);
        g.add(l);
        const cap = label(new THREE.Mesh(new THREE.SphereGeometry(0.03, 16, 10), M.brass()), 'Brass finials');
        cap.position.set(x, h + 0.01, z);
        g.add(cap);
      }
    for (const z of [-w / 2, w / 2]) {
      const r = rbox(len, 0.06, 0.05, wood, 0.015);
      r.position.set(0, h - 0.04, z);
      g.add(r);
    }
    for (const x of [-len / 2, len / 2]) {
      const r = railZ(w, 0.06, 0.05, wood, 0.015);
      r.position.set(x, h - 0.04, 0);
      g.add(r);
    }
    const weave = label(new THREE.Mesh(scaleUV(new THREE.PlaneGeometry(len - 0.05, w - 0.05), len, w), M.niwar()), 'Hand-woven niwar');
    weave.rotation.x = -Math.PI / 2;
    weave.position.y = h - 0.03;
    g.add(weave);
    const quilt = label(rbox(0.6, 0.07, w - 0.12, M.rug(0.6, w - 0.12), 0.02), 'Block-printed quilt');
    quilt.position.set(len / 2 - 0.42, h + 0.0, 0);
    g.add(quilt);
    const b1 = label(bolster(w - 0.2, 0.09), 'Indigo bolster');
    b1.rotation.set(0, Math.PI / 2, Math.PI / 2);
    b1.position.set(-len / 2 + 0.16, h + 0.06, 0);
    g.add(b1);
    const c1 = cushion(0.42, 0.4, 0.13, rust());
    c1.position.set(-len / 2 + 0.3, h + 0.2, 0.15);
    c1.rotation.set(0.1, Math.PI / 2 - 0.2, -0.35);
    g.add(c1);
    return shadowed(g);
  }

  /** Chowki low table in mango wood with brass inlay. */
  function chowki(size = 0.82, h = 0.34) {
    const g = new THREE.Group();
    const wood = M.wood('mango', 23);
    const top = label(rbox(size, 0.05, size, wood, 0.01), 'Mango-wood top');
    top.position.y = h - 0.025;
    g.add(top);
    const inset = 0.06;
    for (const s of [-1, 1]) {
      const a = label(box(size - inset * 2, 0.004, 0.008, M.brass()), 'Brass inlay');
      a.position.set(0, h + 0.001, s * (size / 2 - inset));
      const b = box(0.008, 0.004, size - inset * 2, M.brass());
      b.position.set(s * (size / 2 - inset), h + 0.001, 0);
      g.add(a, b);
    }
    const apron = 0.07;
    for (const s of [-1, 1]) {
      const a = rbox(size - 0.08, apron, 0.025, wood);
      a.position.set(0, h - 0.05 - apron / 2, s * (size / 2 - 0.04));
      const b = railZ(size - 0.08, apron, 0.025, wood);
      b.position.set(s * (size / 2 - 0.04), h - 0.05 - apron / 2, 0);
      g.add(a, b);
    }
    const legGeo = turnedLeg(h - 0.05, 0.03, 0.036, [{ at: 0.3, r: 0.045, w: 0.06 }, { at: 0.7, r: 0.042, w: 0.05 }]);
    for (const x of [-1, 1])
      for (const z of [-1, 1]) {
        const l = label(new THREE.Mesh(legGeo, wood), 'Turned legs');
        l.position.set(x * (size / 2 - 0.05), 0, z * (size / 2 - 0.05));
        g.add(l);
        const corner = box(0.07, 0.005, 0.07, M.brass());
        corner.position.set(x * (size / 2 - 0.035), h + 0.0015, z * (size / 2 - 0.035));
        g.add(corner);
      }
    const urli = label(vase('bowl', M.brass(), 1.6), 'Brass urli');
    urli.position.set(-0.1, h, 0.05);
    const v = vase('round', M.terracotta(), 0.8);
    v.position.set(0.2, h, -0.15);
    g.add(urli, v);
    return shadowed(g);
  }

  /** Teak console with twin jaali doors. Long axis X, faces +Z. */
  function jaaliConsole(len = 1.35, d = 0.4, h = 0.8) {
    const g = new THREE.Group();
    const wood = M.wood('teak', 29);
    const legH = 0.16;
    const bodyH = h - legH;
    const top = label(rbox(len, 0.035, d, wood, 0.006), 'Solid teak top');
    top.position.y = h - 0.0175;
    const bottom = rbox(len - 0.04, 0.03, d - 0.02, wood);
    bottom.position.y = legH + 0.015;
    const backP = box(len - 0.04, bodyH, 0.015, M.colour('#3a2a20', 0.8));
    backP.position.set(0, legH + bodyH / 2, -d / 2 + 0.01);
    g.add(top, bottom, backP);
    for (const s of [-1, 1]) {
      const side = rbox(0.03, bodyH, d - 0.02, wood);
      side.position.set(s * (len / 2 - 0.02), legH + bodyH / 2, 0);
      g.add(side);
    }
    const mid = rbox(0.03, bodyH - 0.04, d - 0.04, wood);
    mid.position.set(0, legH + bodyH / 2, 0);
    g.add(mid);
    // Doors: frame + jaali panel
    const dw = (len - 0.07) / 2;
    const dh = bodyH - 0.06;
    for (const s of [-1, 1]) {
      const door = label(new THREE.Group(), 'Hand-cut jaali doors');
      const fr = 0.045;
      const t = rbox(dw, fr, 0.022, wood);
      t.position.y = dh / 2 - fr / 2;
      const b = rbox(dw, fr, 0.022, wood);
      b.position.y = -dh / 2 + fr / 2;
      const l = rbox(fr, dh, 0.022, wood);
      l.position.x = -dw / 2 + fr / 2;
      const r = rbox(fr, dh, 0.022, wood);
      r.position.x = dw / 2 - fr / 2;
      const jaali = new THREE.Mesh(
        scaleUV(new THREE.PlaneGeometry(dw - fr * 1.5, dh - fr * 1.5), dw, dh),
        M.woodAlpha('teak', T.jaaliAlpha(512, 4), 0.22),
      );
      door.add(t, b, l, r, jaali);
      const knob = new THREE.Mesh(new THREE.SphereGeometry(0.014, 16, 12), M.brass());
      knob.position.set(-s * (dw / 2 - 0.06), 0, 0.02);
      door.add(knob);
      door.position.set(s * (dw / 2 + 0.008), legH + bodyH / 2, d / 2 - 0.012);
      g.add(door);
    }
    for (const x of [-1, 1])
      for (const z of [-1, 1]) {
        const leg = label(new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.014, legH, 16), wood), 'Tapered legs, brass shoes');
        leg.position.set(x * (len / 2 - 0.06), legH / 2, z * (d / 2 - 0.06));
        const f = new THREE.Mesh(new THREE.CylinderGeometry(0.0145, 0.014, 0.03, 16), M.brass());
        f.position.set(x * (len / 2 - 0.06), 0.015, z * (d / 2 - 0.06));
        g.add(leg, f);
      }
    return shadowed(g);
  }

  /** Mango wood sideboard with reeded doors and brass pulls. */
  function reededSideboard(len = 1.8, d = 0.45, h = 0.78) {
    const g = new THREE.Group();
    const wood = M.wood('mango', 31);
    const legH = 0.12;
    const bodyH = h - legH;
    const body = label(rbox(len, bodyH, d - 0.03, wood, 0.01), 'Carcass');
    body.position.set(0, legH + bodyH / 2, -0.015);
    const top = label(rbox(len + 0.02, 0.035, d + 0.01, wood, 0.008), 'Mango-wood top');
    top.position.y = h;
    g.add(body, top);
    const doors = 4;
    const dw = (len - 0.06) / doors;
    const reed = new THREE.CylinderGeometry(0.011, 0.011, bodyH - 0.06, 10, 1, false, 0, Math.PI);
    reed.rotateY(-Math.PI / 2);
    projectUV(reed);
    const perDoor = Math.floor(dw / 0.022);
    const inst = label(new THREE.InstancedMesh(reed, wood, perDoor * doors), 'Reeded doors');
    let k = 0;
    const m4 = new THREE.Matrix4();
    for (let i = 0; i < doors; i++) {
      const x0 = -len / 2 + 0.03 + i * dw;
      for (let j = 0; j < perDoor; j++) {
        m4.makeTranslation(x0 + 0.012 + j * (dw / perDoor), legH + bodyH / 2, d / 2 - 0.03);
        inst.setMatrixAt(k++, m4);
      }
      const pull = label(new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.007, 0.22, 12), M.brass()), 'Brass pulls');
      pull.position.set(x0 + (i % 2 ? 0.05 : dw - 0.05), legH + bodyH / 2, d / 2 - 0.0);
      g.add(pull);
    }
    g.add(inst);
    for (const x of [-1, 1])
      for (const z of [-1, 1]) {
        const leg = rbox(0.05, legH, 0.05, wood);
        leg.position.set(x * (len / 2 - 0.08), legH / 2, z * (d / 2 - 0.08));
        g.add(leg);
      }
    return shadowed(g);
  }

  /** Jhoola: a hanging swing bench on brass chains that rise out of frame. */
  function jhoola(len = 1.25, chain = 2.4) {
    const g = new THREE.Group();
    const wood = M.wood('sheesham', 37);
    const seatY = 0.48;
    const d = 0.58;
    const seat = label(rbox(len, 0.05, d, wood, 0.012), 'Sheesham seat');
    seat.position.y = seatY;
    g.add(seat);
    // Low spindle back and arms
    const backRail = rbox(len, 0.05, 0.05, wood, 0.012);
    backRail.position.set(0, seatY + 0.36, -d / 2 + 0.03);
    g.add(backRail);
    const spindleGeo = turnedLeg(0.33, 0.012, 0.012, [{ at: 0.5, r: 0.02, w: 0.08 }], 14);
    for (let i = 0; i <= 12; i++) {
      const s = label(new THREE.Mesh(spindleGeo, wood), 'Turned spindles');
      s.position.set(-len / 2 + 0.04 + (i * (len - 0.08)) / 12, seatY + 0.025, -d / 2 + 0.03);
      g.add(s);
    }
    for (const sx of [-1, 1]) {
      const arm = railZ(d, 0.045, 0.05, wood, 0.012);
      arm.position.set(sx * (len / 2 - 0.03), seatY + 0.24, 0);
      g.add(arm);
      for (let i = 0; i < 4; i++) {
        const s = new THREE.Mesh(turnedLeg(0.21, 0.011, 0.011, [{ at: 0.5, r: 0.018, w: 0.1 }], 12), wood);
        s.position.set(sx * (len / 2 - 0.03), seatY + 0.025, -d / 2 + 0.06 + (i * (d - 0.1)) / 3);
        g.add(s);
      }
    }
    const c = label(cushion(len - 0.1, 0.07, d - 0.08, M.fabric([233, 224, 207])), 'Cotton cushions');
    c.position.set(0, seatY + 0.06, 0.02);
    g.add(c);
    const b = bolster(0.5, 0.08, M.fabric([52, 62, 94]));
    b.position.set(-0.25, seatY + 0.17, -d / 2 + 0.12);
    g.add(b);
    const p = cushion(0.36, 0.36, 0.11, M.fabric([170, 96, 70]));
    p.position.set(0.3, seatY + 0.26, -d / 2 + 0.12);
    p.rotation.set(-0.2, -0.15, 0.06);
    g.add(p);
    // Brass chains (torus links)
    const link = new THREE.TorusGeometry(0.018, 0.0045, 6, 14);
    const n = Math.ceil(chain / 0.03);
    const corners = [
      [-len / 2 + 0.04, d / 2 - 0.05],
      [len / 2 - 0.04, d / 2 - 0.05],
      [-len / 2 + 0.04, -d / 2 + 0.03],
      [len / 2 - 0.04, -d / 2 + 0.03],
    ];
    const chainMesh = label(new THREE.InstancedMesh(link, M.brass(), n * corners.length), 'Brass chains');
    const m4 = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    let k = 0;
    for (const [cx, cz] of corners) {
      const y0 = cz > 0 ? seatY + 0.03 : seatY + 0.39;
      for (let i = 0; i < n; i++) {
        q.setFromEuler(new THREE.Euler(0, i % 2 ? Math.PI / 2 : 0, Math.PI / 2));
        m4.compose(new THREE.Vector3(cx, y0 + i * 0.03 + 0.02, cz), q, new THREE.Vector3(1, 1, 1));
        chainMesh.setMatrixAt(k++, m4);
      }
    }
    g.add(chainMesh);
    return shadowed(g);
  }

  /** Mudha stool: hourglass reed body with a cushioned top. */
  function mudha(h = 0.42) {
    const g = new THREE.Group();
    const body = label(lathe(
      [
        [0, 0.0],
        [0.2, 0.0],
        [0.19, 0.03],
        [0.13, h * 0.5],
        [0.18, h - 0.04],
        [0.19, h - 0.02],
        [0, h - 0.02],
      ],
      M.reed(),
      48,
    ), 'Sarkanda reed body');
    const geo = body.geometry;
    swapLatheUV(geo, 1.3, 0.45);
    g.add(body);
    for (const [y, r] of [
      [0.012, 0.2],
      [h - 0.03, 0.19],
    ]) {
      const ring = label(new THREE.Mesh(new THREE.TorusGeometry(r, 0.012, 10, 48), M.rope()), 'Jute rope binding');
      ring.rotation.x = Math.PI / 2;
      ring.position.y = y;
      g.add(ring);
    }
    const top = label(new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.185, 0.05, 48), M.fabric([170, 96, 70])), 'Cotton cushion top');
    top.position.y = h + 0.0;
    g.add(top);
    return shadowed(g);
  }


  return { vase, terracottaPot, snakePlant, leafyPlant, tree, shrub, palm, linen, sandLinen, indigo, rust, cushion, bolster, sofa, coffeeTable, bookStack, pendant, floorLamp, artwork, diningTable, diningChair, charpai, chowki, jaaliConsole, reededSideboard, jhoola, mudha };
}

export type Furniture = ReturnType<typeof createFurniture>;

function swapLatheUV(geo: THREE.BufferGeometry, su: number, sv: number) {
  const uv = geo.getAttribute('uv');
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * su, uv.getY(i) * sv);
}

export function mulberry(seed: number) {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
