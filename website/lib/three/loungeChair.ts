import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import { archPoints, archShape, projectUV, scaleUV, shadowed, turnedLeg } from './geometry';
import { caneTexture, fabricTexture, woodTexture, type WoodSpecies } from './textures';

/**
 * "Jharokha" lounge chair: a turned-leg wooden frame, hand-woven cane
 * seat, side and arched back panels, brass ferrules and a linen cushion.
 * Built entirely from code so the viewer needs no model download.
 * Dimensions are in metres; the chair faces +Z with its feet at y = 0.
 */

export interface ChairMaterials {
  wood: THREE.MeshStandardMaterial;
  cane: THREE.MeshStandardMaterial;
  brass: THREE.MeshStandardMaterial;
  cushion: THREE.MeshStandardMaterial;
  bolster: THREE.MeshStandardMaterial;
}

const WOOD_REPEAT = 1 / 0.42; // one texture tile ≈ 42 cm of timber
const CANE_REPEAT = 1 / 0.1; // one tile ≈ 10 cm of webbing

export function createChairMaterials(species: WoodSpecies = 'sheesham'): ChairMaterials {
  const woodMap = woodTexture(species);
  woodMap.repeat.set(WOOD_REPEAT, WOOD_REPEAT);
  const caneMap = caneTexture();
  caneMap.repeat.set(CANE_REPEAT, CANE_REPEAT);
  const linen = fabricTexture([233, 223, 205]);
  linen.repeat.set(4, 4);
  const indigo = fabricTexture([58, 70, 104]);
  indigo.repeat.set(3, 3);
  return {
    wood: new THREE.MeshStandardMaterial({ map: woodMap, roughness: 0.48, metalness: 0 }),
    cane: new THREE.MeshStandardMaterial({
      map: caneMap,
      roughness: 0.62,
      alphaTest: 0.5,
      side: THREE.DoubleSide,
    }),
    brass: new THREE.MeshStandardMaterial({ color: '#c09a5e', metalness: 1, roughness: 0.32 }),
    cushion: new THREE.MeshStandardMaterial({ map: linen, roughness: 0.92 }),
    bolster: new THREE.MeshStandardMaterial({ map: indigo, roughness: 0.9 }),
  };
}

/** Swap the timber finish in place (used by the viewer's material picker). */
export function setWoodSpecies(mats: ChairMaterials, species: WoodSpecies) {
  const next = woodTexture(species);
  next.repeat.set(WOOD_REPEAT, WOOD_REPEAT);
  mats.wood.map?.dispose();
  mats.wood.map = next;
  mats.wood.needsUpdate = true;
}

function rail(len: number, h: number, d: number, mat: THREE.Material, r = 0.006) {
  const geo = projectUV(new RoundedBoxGeometry(len, h, d, 3, r));
  return new THREE.Mesh(geo, mat);
}

/** Cane panel in the XY plane, UVs in metres. */
function canePanel(w: number, h: number, mat: THREE.Material) {
  const geo = scaleUV(new THREE.PlaneGeometry(w, h), w, h);
  return new THREE.Mesh(geo, mat);
}

export function createLoungeChair(mats: ChairMaterials) {
  const chair = new THREE.Group();
  chair.name = 'JharokhaLoungeChair';

  const halfW = 0.33;
  const front = 0.3;
  const back = -0.29;
  const armH = 0.6;
  const seatH = 0.38;

  // Turned legs with a bead below the arm, finished with brass ferrules
  const legGeo = turnedLeg(armH, 0.017, 0.023, [
    { at: 0.84, r: 0.027, w: 0.035 },
    { at: 0.12, r: 0.022, w: 0.03 },
  ]);
  const ferruleGeo = new THREE.CylinderGeometry(0.0195, 0.0185, 0.045, 24);
  for (const x of [-halfW, halfW]) {
    for (const z of [front, back]) {
      const leg = new THREE.Mesh(legGeo, mats.wood);
      leg.position.set(x, 0, z);
      chair.add(leg);
      const ferrule = new THREE.Mesh(ferruleGeo, mats.brass);
      ferrule.position.set(x, 0.0225, z);
      chair.add(ferrule);
    }
  }

  // Arm rests: long along Z, so build along X and turn
  for (const x of [-halfW, halfW]) {
    const arm = rail(0.7, 0.026, 0.074, mats.wood, 0.011);
    arm.rotation.y = Math.PI / 2;
    arm.position.set(x, armH + 0.013, 0.005);
    chair.add(arm);
    const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 0.006, 20), mats.brass);
    cap.position.set(x, armH + 0.028, front);
    chair.add(cap);
  }

  // Seat frame
  const seatW = halfW * 2;
  const frontRail = rail(seatW, 0.05, 0.032, mats.wood);
  frontRail.position.set(0, seatH - 0.02, front);
  const backRail = rail(seatW, 0.05, 0.032, mats.wood);
  backRail.position.set(0, seatH - 0.02, back);
  chair.add(frontRail, backRail);
  for (const x of [-halfW, halfW]) {
    const side = rail(front - back, 0.05, 0.032, mats.wood);
    side.rotation.y = Math.PI / 2;
    side.position.set(x, seatH - 0.02, (front + back) / 2);
    chair.add(side);
    // Low stretcher
    const str = rail(front - back, 0.022, 0.022, mats.wood, 0.008);
    str.rotation.y = Math.PI / 2;
    str.position.set(x, 0.13, (front + back) / 2);
    chair.add(str);
    // Cane side panel between seat rail and arm
    const sp = canePanel(front - back - 0.03, armH - seatH - 0.02, mats.cane);
    sp.rotation.y = Math.PI / 2;
    sp.position.set(x, (armH + seatH) / 2, (front + back) / 2);
    chair.add(sp);
  }
  const seat = canePanel(seatW - 0.03, front - back - 0.03, mats.cane);
  seat.rotation.x = -Math.PI / 2;
  seat.position.set(0, seatH - 0.004, (front + back) / 2);
  chair.add(seat);

  // Reclined arched back: a bent-wood tube framing an arched cane panel
  const backGroup = new THREE.Group();
  const aHalf = 0.27;
  const spring = 0.28;
  const pts = archPoints(aHalf, spring, 40).map((p) => new THREE.Vector3(p.x, p.y, 0));
  const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
  const tubeGeo = new THREE.TubeGeometry(curve, 160, 0.016, 14, false);
  // Tube UVs run along the curve in U: scale to metres for consistent grain
  scaleUV(tubeGeo, curve.getLength(), 0.1);
  backGroup.add(new THREE.Mesh(tubeGeo, mats.wood));
  const bottom = rail(aHalf * 2 + 0.03, 0.036, 0.032, mats.wood);
  bottom.position.y = 0.012;
  backGroup.add(bottom);
  const panelGeo = new THREE.ShapeGeometry(archShape(aHalf - 0.008, spring, 40), 1);
  backGroup.add(new THREE.Mesh(panelGeo, mats.cane));
  // Brass finial at the crown of the arch
  const finial = new THREE.Mesh(new THREE.SphereGeometry(0.016, 20, 14), mats.brass);
  finial.position.set(0, spring + aHalf + 0.02, 0);
  backGroup.add(finial);
  backGroup.position.set(0, seatH + 0.005, back - 0.005);
  backGroup.rotation.x = -0.24;
  chair.add(backGroup);

  // Linen seat cushion and an indigo bolster
  const cushion = new THREE.Mesh(projectUV(new RoundedBoxGeometry(seatW - 0.04, 0.085, front - back - 0.05, 5, 0.035)), mats.cushion);
  cushion.position.set(0, seatH + 0.045, (front + back) / 2 + 0.02);
  chair.add(cushion);
  const bolster = new THREE.Mesh(new THREE.CapsuleGeometry(0.07, 0.36, 10, 24), mats.bolster);
  bolster.rotation.z = Math.PI / 2;
  bolster.position.set(0, seatH + 0.15, back + 0.09);
  bolster.scale.set(1, 1, 0.82);
  chair.add(bolster);

  return shadowed(chair);
}
