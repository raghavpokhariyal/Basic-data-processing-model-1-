import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { createChairMaterials, createLoungeChair, setWoodSpecies } from './loungeChair';
import type { WoodSpecies } from './textures';

export interface ViewerApi {
  setSpecies: (s: WoodSpecies) => void;
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

/**
 * Interactive viewer for the Jharokha chair: orbit to rotate, pinch or
 * Ctrl/⌘ + scroll to zoom (plain scrolling keeps scrolling the page),
 * and only renders while on screen.
 */
export function createViewer(container: HTMLElement, opts: { species: WoodSpecies; reducedMotion: boolean; onReady?: () => void }): ViewerApi {
  const coarse = window.matchMedia('(pointer: coarse)').matches;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, coarse ? 1.5 : 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.domElement.setAttribute('aria-hidden', 'true');
  container.appendChild(renderer.domElement);

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
  sc.far = 8;
  scene.add(key);
  const fill = new THREE.DirectionalLight('#dfe6f2', 0.4);
  fill.position.set(3, 1.5, 2);
  scene.add(fill);

  const ground = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.ShadowMaterial({ opacity: 0.16 }));
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);
  const blob = new THREE.Mesh(new THREE.PlaneGeometry(1.5, 1.5), new THREE.MeshBasicMaterial({ map: blobShadow(), transparent: true, depthWrite: false }));
  blob.rotation.x = -Math.PI / 2;
  blob.position.y = 0.001;
  scene.add(blob);

  const mats = createChairMaterials(opts.species);
  const chair = createLoungeChair(mats);
  chair.rotation.y = -0.5;
  scene.add(chair);

  const camera = new THREE.PerspectiveCamera(30, 1, 0.05, 30);
  const home = new THREE.Vector3(1.85, 1.2, 2.8);
  camera.position.copy(home);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.42, -0.02);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.minDistance = 1.3;
  controls.maxDistance = 3.6;
  controls.minPolarAngle = 0.35;
  controls.maxPolarAngle = 1.48;
  controls.rotateSpeed = 0.7;
  controls.autoRotate = !opts.reducedMotion;
  controls.autoRotateSpeed = 0.8;
  controls.update();
  // Let vertical swipes keep scrolling the page on phones; horizontal drags rotate
  renderer.domElement.style.touchAction = 'pan-y';

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
    if (opts.reducedMotion) return;
    window.clearTimeout(idle);
    idle = window.setTimeout(() => (controls.autoRotate = true), 5000);
  });

  const resize = () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    camera.aspect = w / h;
    // Keep the chair framed on narrow (portrait) containers
    camera.fov = w / h < 0.9 ? 38 : 30;
    camera.updateProjectionMatrix();
  };
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();

  let running = false;
  const loop = () => {
    controls.update();
    renderer.render(scene, camera);
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
  loop();
  opts.onReady?.();

  const spherical = new THREE.Spherical();
  const offset = new THREE.Vector3();
  return {
    setSpecies: (s) => setWoodSpecies(mats, s),
    rotate: (rad) => {
      controls.autoRotate = false;
      offset.copy(camera.position).sub(controls.target);
      spherical.setFromVector3(offset);
      spherical.theta += rad;
      offset.setFromSpherical(spherical);
      camera.position.copy(controls.target).add(offset);
      controls.update();
    },
    zoom: (factor) => {
      offset.copy(camera.position).sub(controls.target);
      const d = THREE.MathUtils.clamp(offset.length() * factor, controls.minDistance, controls.maxDistance);
      offset.setLength(d);
      camera.position.copy(controls.target).add(offset);
      controls.update();
    },
    reset: () => {
      camera.position.copy(home);
      controls.update();
    },
    dispose: () => {
      window.clearTimeout(idle);
      io.disconnect();
      ro.disconnect();
      renderer.setAnimationLoop(null);
      container.removeEventListener('wheel', gate, { capture: true });
      container.removeEventListener('pointerdown', gate, { capture: true });
      controls.dispose();
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.isMesh) {
          m.geometry.dispose();
          const list = Array.isArray(m.material) ? m.material : [m.material];
          list.forEach((mat) => {
            (mat as THREE.MeshStandardMaterial).map?.dispose();
            mat.dispose();
          });
        }
      });
      scene.environment?.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
