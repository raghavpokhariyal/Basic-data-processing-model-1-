import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { gsap } from 'gsap';
import { createLoungeChair } from './loungeChair';
import { createFurniture } from './furniture';
import { createKit } from './kit';
import type { WoodSpecies } from './textures';

export type PieceId = 'lounge' | 'dining' | 'charpai' | 'chowki' | 'console' | 'sideboard' | 'jhoola' | 'mudha';

/** The timber each piece is made in by default (the mudha has none). */
export const PIECE_SPECIES: Record<PieceId, WoodSpecies | null> = {
  lounge: 'sheesham',
  dining: 'sheesham',
  charpai: 'sheesham',
  chowki: 'mango',
  console: 'teak',
  sideboard: 'mango',
  jhoola: 'sheesham',
  mudha: null,
};

export interface ViewerApi {
  setPiece: (p: PieceId) => void;
  setSpecies: (s: WoodSpecies) => void;
  setExploded: (on: boolean) => void;
  rotate: (radians: number) => void;
  zoom: (factor: number) => void;
  reset: () => void;
  dispose: () => void;
}

/** Soft radial contact shadow painted on a canvas. */
function blobShadow() {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const ctx = c.getContext('2d')!;
  const g = ctx.createRadialGradient(128, 128, 10, 128, 128, 128);
  g.addColorStop(0, 'rgba(29,37,57,0.42)');
  g.addColorStop(0.55, 'rgba(29,37,57,0.12)');
  g.addColorStop(1, 'rgba(29,37,57,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 256, 256);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

const VIEW_DIR = new THREE.Vector3(1.15, 0.62, 1.6).normalize();

/**
 * Interactive viewer for the workshop's furniture. Orbit to rotate, pinch
 * or Ctrl/⌘ + scroll to zoom (plain scrolling keeps scrolling the page),
 * switch pieces and timber, or pull a piece apart into a labelled
 * exploded view. Renders only while on screen.
 */
export function createViewer(
  container: HTMLElement,
  opts: { piece?: PieceId; species: WoodSpecies; reducedMotion: boolean; onReady?: () => void },
): ViewerApi {
  const coarse = window.matchMedia('(pointer: coarse)').matches;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, coarse ? 1.5 : 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.domElement.setAttribute('aria-hidden', 'true');
  container.prepend(renderer.domElement);

  const labelLayer = document.createElement('div');
  labelLayer.setAttribute('aria-hidden', 'true');
  labelLayer.style.cssText = 'position:absolute;inset:0;pointer-events:none;';
  container.append(labelLayer);

  const scene = new THREE.Scene();
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.55;
  pmrem.dispose();

  const key = new THREE.DirectionalLight('#fff1de', 2.4);
  key.position.set(-2.2, 3.4, 2.4);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.radius = 6;
  key.shadow.bias = -0.0005;
  const sc = key.shadow.camera;
  sc.left = sc.bottom = -2.4;
  sc.right = sc.top = 2.4;
  sc.near = 0.5;
  sc.far = 9;
  scene.add(key);
  const fill = new THREE.DirectionalLight('#dfe6f2', 0.4);
  fill.position.set(3, 1.5, 2);
  scene.add(fill);

  const ground = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), new THREE.ShadowMaterial({ opacity: 0.16 }));
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);
  const blob = new THREE.Mesh(new THREE.PlaneGeometry(1.6, 1.6), new THREE.MeshBasicMaterial({ map: blobShadow(), transparent: true, depthWrite: false }));
  blob.rotation.x = -Math.PI / 2;
  blob.position.y = 0.001;
  scene.add(blob);

  const M = createKit();
  const F = createFurniture(M);
  const chairMats = { wood: M.wood('sheesham', 7), cane: M.cane(), brass: M.brass(), cushion: M.fabric([233, 223, 205]), bolster: M.fabric([58, 70, 104]) };
  const build: Record<PieceId, () => THREE.Object3D> = {
    lounge: () => createLoungeChair(chairMats),
    dining: () => F.diningChair('sheesham'),
    charpai: () => F.charpai(),
    chowki: () => F.chowki(),
    console: () => F.jaaliConsole(),
    sideboard: () => F.reededSideboard(),
    jhoola: () => F.jhoola(1.25, 1.1),
    mudha: () => F.mudha(),
  };
  let species = opts.species;

  const camera = new THREE.PerspectiveCamera(30, 1, 0.05, 40);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.minPolarAngle = 0.3;
  controls.maxPolarAngle = 1.5;
  controls.rotateSpeed = 0.7;
  controls.autoRotate = !opts.reducedMotion;
  controls.autoRotateSpeed = 0.8;
  renderer.domElement.style.touchAction = 'pan-y';
  const dur = (s: number) => (opts.reducedMotion ? 0 : s);

  // Zoom only on pinch / Ctrl+wheel so normal scrolling still moves the page
  const gate = (e: Event) => {
    if (e.type === 'wheel') {
      const w = e as WheelEvent;
      controls.enableZoom = w.ctrlKey || w.metaKey;
    } else if ((e as PointerEvent).pointerType === 'touch') {
      controls.enableZoom = true;
    }
  };
  container.addEventListener('wheel', gate, { capture: true, passive: true });
  container.addEventListener('pointerdown', gate, { capture: true });

  let idle: number | undefined;
  controls.addEventListener('start', () => {
    controls.autoRotate = false;
    window.clearTimeout(idle);
  });
  controls.addEventListener('end', () => {
    if (opts.reducedMotion || exploded) return;
    window.clearTimeout(idle);
    idle = window.setTimeout(() => (controls.autoRotate = true), 5000);
  });

  // Current piece and its exploded layout
  let piece: THREE.Object3D | null = null;
  let parts: { obj: THREE.Object3D; home: THREE.Vector3; offset: THREE.Vector3; label?: string }[] = [];
  let labels: { el: HTMLElement; part: THREE.Object3D }[] = [];
  let exploded = false;
  let home = new THREE.Vector3();
  let radius = 1;
  const target = new THREE.Vector3();

  const fit = (animate: boolean) => {
    const vFov = THREE.MathUtils.degToRad(camera.fov / 2);
    const hFov = Math.atan(Math.tan(vFov) * camera.aspect);
    const dist = (radius * (exploded ? 1.55 : 1.12)) / Math.sin(Math.min(vFov, hFov));
    controls.minDistance = dist * 0.5;
    controls.maxDistance = dist * 1.8;
    const dir = camera.position.clone().sub(controls.target);
    const keepAngle = animate && dir.lengthSq() > 0.0001;
    const to = target.clone().addScaledVector(keepAngle ? dir.normalize() : VIEW_DIR, dist);
    home = target.clone().addScaledVector(VIEW_DIR, dist);
    if (animate && !opts.reducedMotion) {
      gsap.to(camera.position, { x: to.x, y: to.y, z: to.z, duration: 0.9, ease: 'power3.inOut' });
      gsap.to(controls.target, { x: target.x, y: target.y, z: target.z, duration: 0.9, ease: 'power3.inOut' });
    } else {
      camera.position.copy(to);
      controls.target.copy(target);
    }
  };

  function clearLabels() {
    labels.forEach((l) => l.el.remove());
    labels = [];
  }

  function setPiece(id: PieceId) {
    if (piece) {
      scene.remove(piece);
      piece.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.isMesh) m.geometry.dispose();
      });
    }
    clearLabels();
    piece = build[id]();
    piece.rotation.y = -0.5;
    scene.add(piece);
    const native = PIECE_SPECIES[id];
    if (native) {
      species = native;
      M.setWoodSpecies(native);
    }
    piece.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(piece);
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    radius = sphere.radius;
    target.copy(sphere.center);
    blob.scale.setScalar(Math.max(box.max.x - box.min.x, box.max.z - box.min.z) / 1.1);

    // Exploded offsets: each part moves away from the centre of the piece
    const centre = sphere.center;
    const seen = new Set<string>();
    parts = piece.children.map((obj) => {
      const b = new THREE.Box3().setFromObject(obj);
      const c = b.getCenter(new THREE.Vector3());
      const d = c.clone().sub(centre);
      d.y *= 1.6;
      if (d.lengthSq() < 1e-4) d.set(0, 1, 0);
      const dist = (0.1 + d.length() * 0.75) * (radius / 0.6);
      const offset = d.normalize().multiplyScalar(Math.min(dist, radius * 0.9));
      // Convert the world-space offset into the piece's local frame
      const inv = new THREE.Quaternion().copy(piece!.quaternion).invert();
      offset.applyQuaternion(inv);
      const label = obj.userData.label as string | undefined;
      if (label && !seen.has(label)) {
        seen.add(label);
        const el = document.createElement('div');
        el.className = 'pointer-events-none absolute top-0 left-0 opacity-0 transition-opacity duration-500';
        el.innerHTML = `<span class="flex -translate-x-1/2 -translate-y-1/2 items-center gap-1.5 rounded-full bg-indigo/92 px-2.5 py-1 text-[0.66rem] font-semibold whitespace-nowrap text-ivory shadow-sm"><span class="h-1.5 w-1.5 rounded-full bg-brass-light"></span>${label}</span>`;
        labelLayer.append(el);
        labels.push({ el, part: obj });
      }
      return { obj, home: obj.position.clone(), offset, label };
    });
    if (exploded) applyExplode(false);
    fit(true);
  }

  function applyExplode(animate: boolean) {
    parts.forEach((p, i) => {
      const to = exploded ? p.home.clone().add(p.offset) : p.home;
      gsap.killTweensOf(p.obj.position);
      if (animate && !opts.reducedMotion) gsap.to(p.obj.position, { x: to.x, y: to.y, z: to.z, duration: 0.9, delay: i * 0.012, ease: 'power3.inOut' });
      else p.obj.position.copy(to);
    });
    labels.forEach((l) => {
      l.el.style.opacity = exploded ? '1' : '0';
    });
    controls.autoRotate = !exploded && !opts.reducedMotion;
    fit(true);
  }

  const resize = () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    camera.aspect = w / h;
    camera.fov = w / h < 0.9 ? 36 : 30;
    camera.updateProjectionMatrix();
  };
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();

  const v = new THREE.Vector3();
  const box = new THREE.Box3();
  const placeLabels = () => {
    if (!exploded) return;
    for (const l of labels) {
      box.setFromObject(l.part).getCenter(v);
      v.project(camera);
      const x = (v.x * 0.5 + 0.5) * container.clientWidth;
      const y = (-v.y * 0.5 + 0.5) * container.clientHeight;
      l.el.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0)`;
    }
  };

  let running = false;
  const loop = () => {
    controls.update();
    renderer.render(scene, camera);
    placeLabels();
  };
  const io = new IntersectionObserver(([e]) => {
    if (e.isIntersecting && !running) {
      running = true;
      renderer.setAnimationLoop(loop);
    } else if (!e.isIntersecting && running) {
      running = false;
      renderer.setAnimationLoop(null);
    }
  });
  io.observe(container);

  setPiece(opts.piece ?? 'lounge');
  fit(false);
  loop();
  opts.onReady?.();

  const spherical = new THREE.Spherical();
  const offset = new THREE.Vector3();
  return {
    setPiece,
    setSpecies: (s) => {
      species = s;
      M.setWoodSpecies(s);
    },
    setExploded: (on) => {
      exploded = on;
      applyExplode(true);
    },
    rotate: (rad) => {
      controls.autoRotate = false;
      offset.copy(camera.position).sub(controls.target);
      spherical.setFromVector3(offset);
      spherical.theta += rad;
      offset.setFromSpherical(spherical);
      const to = controls.target.clone().add(offset);
      gsap.to(camera.position, { x: to.x, y: to.y, z: to.z, duration: dur(0.5), ease: 'power2.out' });
    },
    zoom: (factor) => {
      offset.copy(camera.position).sub(controls.target);
      offset.setLength(THREE.MathUtils.clamp(offset.length() * factor, controls.minDistance, controls.maxDistance));
      const to = controls.target.clone().add(offset);
      gsap.to(camera.position, { x: to.x, y: to.y, z: to.z, duration: dur(0.5), ease: 'power2.out' });
    },
    reset: () => {
      gsap.to(camera.position, { x: home.x, y: home.y, z: home.z, duration: dur(0.8), ease: 'power3.inOut' });
      gsap.to(controls.target, { x: target.x, y: target.y, z: target.z, duration: dur(0.8), ease: 'power3.inOut' });
    },
    dispose: () => {
      window.clearTimeout(idle);
      io.disconnect();
      ro.disconnect();
      renderer.setAnimationLoop(null);
      container.removeEventListener('wheel', gate, { capture: true });
      container.removeEventListener('pointerdown', gate, { capture: true });
      controls.dispose();
      clearLabels();
      labelLayer.remove();
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.isMesh) m.geometry.dispose();
      });
      M.dispose();
      (blob.material as THREE.MeshBasicMaterial).map?.dispose();
      (blob.material as THREE.Material).dispose();
      (ground.material as THREE.Material).dispose();
      scene.environment?.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
