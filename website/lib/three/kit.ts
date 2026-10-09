import * as THREE from 'three';
import { caneCanvas, fabricCanvas, woodCanvas, type RGB, type WoodSpecies } from './textures';
import * as S from './surfaces';

const std = (o: THREE.MeshStandardMaterialParameters) => new THREE.MeshStandardMaterial(o);

/**
 * A per-viewer material library. Each material is created once on first
 * use and every one is disposed together when the viewer unmounts.
 * Wooden materials are tagged so the finish (sheesham, teak, mango) can
 * be swapped across a whole scene.
 */
export function createKit() {
  const cache = new Map<string, THREE.Material>();
  const memo = <T extends THREE.Material>(key: string, make: () => T): T => {
    let m = cache.get(key) as T | undefined;
    if (!m) {
      m = make();
      cache.set(key, m);
    }
    return m;
  };
  const woodMat = (species: WoodSpecies, seed: number, extra: THREE.MeshStandardMaterialParameters = {}) => {
    const m = std({ map: S.tex(woodCanvas(species, 512, seed), 0.42), roughness: 0.5, ...extra });
    m.userData.wood = { species, seed };
    return m;
  };

  const kit = {
    wood: (species: WoodSpecies = 'sheesham', seed = 7) => memo(`wood-${species}-${seed}`, () => woodMat(species, seed)),
    woodAlpha: (species: WoodSpecies, alpha: HTMLCanvasElement, metres: number) =>
      memo(`woodA-${species}-${metres}`, () =>
        woodMat(species, 3, { alphaMap: S.tex(alpha, metres, false), alphaTest: 0.5, roughness: 0.55, side: THREE.DoubleSide }),
      ),
    cane: () => memo('cane', () => std({ map: S.tex(caneCanvas(512, 4), 0.1), alphaTest: 0.5, side: THREE.DoubleSide, roughness: 0.62 })),
    brass: () => memo('brass', () => std({ color: '#c09a5e', metalness: 1, roughness: 0.3 })),
    fabric: (rgb: RGB, metres = 0.25) => memo(`fab-${rgb.join('-')}`, () => std({ map: S.tex(fabricCanvas(rgb, 128, 4), metres), roughness: 0.95 })),
    niwar: () => memo('niwar', () => std({ map: S.tex(S.niwar(256, 16), 0.5), roughness: 0.9 })),
    reed: () => memo('reed', () => std({ map: S.tex(S.reed(256), 0.35), roughness: 0.85 })),
    rug: (w: number, h: number) =>
      memo(`rug-${w}-${h}`, () => {
        const t = S.tex(S.rug(512, 512 * (h / w)), 1);
        t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
        t.repeat.set(1, 1);
        return std({ map: t, roughness: 0.95 });
      }),
    plaster: (rgb: RGB, key = '') => memo(`pl-${rgb.join('-')}${key}`, () => std({ map: S.tex(S.plaster(rgb, 256, 3), 2.4), roughness: 0.92 })),
    concrete: () => memo('concrete', () => std({ map: S.tex(S.cement(256, 9), 1.6), roughness: 0.9, color: '#f0ece4' })),
    terrazzo: () => memo('terrazzo', () => std({ map: S.tex(S.terrazzo([236, 228, 214], 512), 1.2), roughness: 0.35 })),
    kota: () => memo('kota', () => std({ map: S.tex(S.kota(512, 2), 1.2), roughness: 0.38 })),
    sandstone: (courses = 0, rgb: RGB = [214, 188, 148]) =>
      memo(`ss-${courses}-${rgb.join('-')}`, () => std({ map: S.tex(S.sandstone(rgb, 512, 31, courses), courses ? 1.2 : 1.6), roughness: 0.85 })),
    brick: () => memo('brick', () => std({ map: S.tex(S.brick(512, 8), 0.9), roughness: 0.9 })),
    brickJaali: () =>
      memo('brickJaali', () =>
        std({ map: S.tex(S.brick(512, 8), 0.7), alphaMap: S.tex(S.brickJaaliAlpha(512, 8), 0.7, false), alphaTest: 0.5, roughness: 0.9, side: THREE.DoubleSide }),
      ),
    jaaliStone: (rgb: RGB = [236, 228, 212], metres = 0.6) =>
      memo(`jaaliStone-${rgb.join('-')}-${metres}`, () =>
        std({ map: S.tex(S.sandstone(rgb, 256, 5), 0.8), alphaMap: S.tex(S.jaaliAlpha(512, 4), metres, false), alphaTest: 0.5, roughness: 0.8, side: THREE.DoubleSide }),
      ),
    lawn: () => memo('lawn', () => std({ map: S.tex(S.lawn(256), 3), roughness: 1 })),
    earth: () => memo('earth', () => std({ map: S.tex(S.earth(256), 2.5), roughness: 1 })),
    terracotta: () => memo('tc', () => std({ map: S.tex(S.plaster([176, 98, 70], 128, 8, 2), 0.5), roughness: 0.85 })),
    glazeIvory: () => memo('glazeIv', () => std({ color: '#ece4d4', roughness: 0.22 })),
    glazeIndigo: () => memo('glazeIn', () => std({ color: '#33405f', roughness: 0.25 })),
    leaf: () => memo('leaf', () => std({ color: '#5f6e45', roughness: 0.6, side: THREE.DoubleSide })),
    leafDark: () => memo('leafD', () => std({ color: '#4c5a39', roughness: 0.65, side: THREE.DoubleSide })),
    foliage: () => memo('fol', () => std({ color: '#6f7b52', roughness: 0.95 })),
    bark: () => memo('bark', () => std({ color: '#6e5a48', roughness: 0.95 })),
    water: () => memo('water', () => std({ color: '#6d8c90', roughness: 0.05, metalness: 0.3 })),
    glass: () => memo('glass', () => std({ color: '#5a6672', roughness: 0.06, metalness: 0.85, transparent: true, opacity: 0.82 })),
    dark: () => memo('dark', () => std({ color: '#2b2a2a', roughness: 0.8 })),
    white: () => memo('white', () => std({ color: '#f1ece2', roughness: 0.85 })),
    rope: () => memo('rope', () => std({ color: '#bfa57a', roughness: 0.95 })),
    paper: () => memo('paper', () => std({ color: '#efe9dc', roughness: 0.9 })),
    steel: () => memo('steel', () => std({ color: '#6c6f72', roughness: 0.5, metalness: 0.8 })),
    emissive: (color: string, intensity = 3) => memo(`em-${color}-${intensity}`, () => std({ color: '#000', emissive: color, emissiveIntensity: intensity })),
    colour: (color: string, roughness = 0.8) => memo(`c-${color}-${roughness}`, () => std({ color, roughness })),

    /** Re-texture every wooden material with another species, keeping each grain seed. */
    setWoodSpecies(species: WoodSpecies) {
      for (const m of cache.values()) {
        const w = (m as THREE.MeshStandardMaterial).userData.wood as { species: WoodSpecies; seed: number } | undefined;
        if (!w || w.species === species) continue;
        const mat = m as THREE.MeshStandardMaterial;
        const next = S.tex(woodCanvas(species, 512, w.seed), 0.42);
        mat.map?.dispose();
        mat.map = next;
        mat.needsUpdate = true;
        w.species = species;
      }
    },

    dispose() {
      for (const m of cache.values()) {
        for (const v of Object.values(m)) if (v && (v as THREE.Texture).isTexture) (v as THREE.Texture).dispose();
        m.dispose();
      }
      cache.clear();
    },
  };
  return kit;
}

export type Kit = ReturnType<typeof createKit>;
