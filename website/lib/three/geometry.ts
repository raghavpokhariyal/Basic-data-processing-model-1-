import * as THREE from 'three';

/**
 * Re-projects UVs in metres using the dominant normal axis, so a tiled
 * texture keeps the same density on every part. Grain (texture U) runs
 * along local X, so build long wooden parts along X and rotate them.
 */
export function projectUV<T extends THREE.BufferGeometry>(geo: T, scale = 1): T {
  const pos = geo.getAttribute('position');
  if (!geo.getAttribute('normal')) geo.computeVertexNormals();
  const nor = geo.getAttribute('normal');
  const uv = new Float32Array(pos.count * 2);
  for (let i = 0; i < pos.count; i++) {
    const ax = Math.abs(nor.getX(i));
    const ay = Math.abs(nor.getY(i));
    const az = Math.abs(nor.getZ(i));
    let u: number;
    let v: number;
    if (ax >= ay && ax >= az) {
      u = pos.getZ(i);
      v = pos.getY(i);
    } else if (ay >= az) {
      u = pos.getX(i);
      v = pos.getZ(i);
    } else {
      u = pos.getX(i);
      v = pos.getY(i);
    }
    uv[i * 2] = u * scale;
    uv[i * 2 + 1] = v * scale;
  }
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  return geo;
}

/** Multiplies existing UVs, e.g. to turn a 0..1 plane into metres. */
export function scaleUV<T extends THREE.BufferGeometry>(geo: T, su: number, sv: number): T {
  const uv = geo.getAttribute('uv');
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * su, uv.getY(i) * sv);
  uv.needsUpdate = true;
  return geo;
}

/** Swaps U and V so lathe-turned parts get grain along their length. */
export function swapUV<T extends THREE.BufferGeometry>(geo: T, su = 1, sv = 1): T {
  const uv = geo.getAttribute('uv');
  for (let i = 0; i < uv.count; i++) {
    const u = uv.getX(i);
    const v = uv.getY(i);
    uv.setXY(i, v * su, u * sv);
  }
  uv.needsUpdate = true;
  return geo;
}

/**
 * A lathe-turned leg profile (radius, height pairs from floor up) with
 * optional beads, as on traditional Indian charpai and chowki legs.
 */
export function turnedLeg(
  height: number,
  rBottom: number,
  rTop: number,
  beads: { at: number; r: number; w: number }[] = [],
  segments = 28,
) {
  const pts: THREE.Vector2[] = [new THREE.Vector2(0, 0)];
  const steps = 48;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const y = t * height;
    let r = rBottom + (rTop - rBottom) * t;
    for (const b of beads) {
      const dy = (y - b.at * height) / b.w;
      r += (b.r - r) * Math.exp(-dy * dy * 4) * (Math.abs(dy) < 1.4 ? 1 : 0);
    }
    pts.push(new THREE.Vector2(r, y));
  }
  pts.push(new THREE.Vector2(0, height));
  const geo = new THREE.LatheGeometry(pts, segments);
  // U around, V along height -> swap so grain follows the leg
  return swapUV(geo, height, 2 * Math.PI * Math.max(rBottom, rTop));
}

/** Points of a round (jharokha) arch: straight jambs and a semicircle. */
export function archPoints(halfWidth: number, springHeight: number, segments = 32) {
  const pts: THREE.Vector2[] = [new THREE.Vector2(-halfWidth, 0), new THREE.Vector2(-halfWidth, springHeight)];
  for (let i = 1; i < segments; i++) {
    const a = Math.PI - (i / segments) * Math.PI;
    pts.push(new THREE.Vector2(Math.cos(a) * halfWidth, springHeight + Math.sin(a) * halfWidth));
  }
  pts.push(new THREE.Vector2(halfWidth, springHeight), new THREE.Vector2(halfWidth, 0));
  return pts;
}

export function archShape(halfWidth: number, springHeight: number, segments = 32) {
  return new THREE.Shape(archPoints(halfWidth, springHeight, segments));
}

export function shadowed<T extends THREE.Object3D>(obj: T): T {
  obj.traverse((o) => {
    if ((o as THREE.Mesh).isMesh) {
      o.castShadow = true;
      o.receiveShadow = true;
    }
  });
  return obj;
}
