import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { gsap } from 'gsap';
import { createKit } from './kit';
import { buildVilla, ANNOTATIONS, LAYERS } from './villa';
import { createSky, sunAt } from './sky';
import { houseAnnotations } from '../house';

export type ViewMode = 'eye' | 'bird' | 'axo' | 'plan' | 'elevation';
export type Section = 'none' | 'ground' | 'upper';

export interface HouseViewerApi {
  setView: (v: ViewMode) => void;
  setStage: (n: number) => void;
  setExploded: (on: boolean) => void;
  setSection: (s: Section) => void;
  setHour: (h: number, settle?: boolean) => void;
  zoom: (factor: number) => void;
  reset: () => void;
  dispose: () => void;
}

interface Options {
  reducedMotion: boolean;
  /** Absolutely positioned elements the viewer moves over the canvas. */
  annotationEls: (HTMLElement | null)[];
  layerEls: (HTMLElement | null)[];
  onProgress?: (p: number) => void;
}

const EXPLODE_GAP = 4.6;
const CUT = { none: 1e4, ground: 2.0, upper: 5.6 } as const;
/** Layer each annotation belongs to (hidden until that stage is built). */
const ANNOTATION_LAYER = houseAnnotations.map((a) => a.layer);

interface PerspView {
  kind: 'persp';
  fov: number;
  pos: [number, number, number];
  target: [number, number, number];
  polar: [number, number];
  dist: [number, number];
}
interface OrthoView {
  kind: 'ortho';
  dir: [number, number, number];
  target: [number, number, number];
  height: number;
  rotate: boolean;
  polar?: [number, number];
  plan?: boolean;
}

const VIEWS: Record<ViewMode, PerspView | OrthoView> = {
  eye: { kind: 'persp', fov: 38, pos: [13, 1.65, 19], target: [-0.8, 3.0, -2.5], polar: [1.2, 1.62], dist: [10, 34] },
  bird: { kind: 'persp', fov: 32, pos: [24, 20, 28], target: [0, 2, -2.5], polar: [0.15, 1.35], dist: [16, 90] },
  axo: { kind: 'ortho', dir: [1, 1, 1], target: [0, 2.8, -3], height: 24, rotate: true, polar: [0.9553, 0.9553] },
  plan: { kind: 'ortho', dir: [0, 1, 0], target: [0.8, 0, -2.5], height: 23, rotate: false, plan: true },
  elevation: { kind: 'ortho', dir: [0, 0, 1], target: [0.5, 3.6, -3], height: 11, rotate: false },
};

export async function createHouseViewer(container: HTMLElement, opts: Options): Promise<HouseViewerApi> {
  const coarse = window.matchMedia('(pointer: coarse)').matches;
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, coarse ? 1.5 : 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.localClippingEnabled = true;
  const canvas = renderer.domElement;
  canvas.setAttribute('aria-hidden', 'true');
  canvas.style.display = 'block';
  canvas.style.transition = 'opacity 0.35s ease';
  container.prepend(canvas);

  const scene = new THREE.Scene();
  const sky = createSky();
  scene.add(sky.mesh);

  const M = createKit();
  const { root, layers } = await buildVilla(M, opts.onProgress);
  scene.add(root);

  // A section plane shared by every material; parked far above when off
  const cutPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), CUT.none);
  // Elevations stop at the ground line; parked far below otherwise
  const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 1e4);
  const planes = [cutPlane, groundPlane];
  root.traverse((o) => {
    const m = o as THREE.Mesh;
    if (m.isMesh) (m.material as THREE.Material).clippingPlanes = planes;
  });
  // Poché: back faces of cut walls and structure render as a solid fill
  const poche = new THREE.MeshBasicMaterial({ color: '#1d2539', side: THREE.BackSide, clippingPlanes: planes });
  for (const id of ['walls', 'structure'] as const) {
    const layer = layers.find((l) => l.id === id)!;
    const extra: THREE.Mesh[] = [];
    layer.group.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh && !(o as THREE.InstancedMesh).isInstancedMesh && (m.material as THREE.Material) !== M.glass()) {
        const p = new THREE.Mesh(m.geometry, poche);
        p.matrix.copy(m.matrix);
        p.matrixAutoUpdate = false;
        extra.push(p);
      }
    });
    extra.forEach((p) => layer.group.children[0].add(p));
  }

  // Sun and sky
  const sun = new THREE.DirectionalLight('#fff', 3);
  sun.castShadow = true;
  sun.shadow.mapSize.set(coarse ? 1024 : 2048, coarse ? 1024 : 2048);
  const sc = sun.shadow.camera;
  sc.left = sc.bottom = -17;
  sc.right = sc.top = 17;
  sc.near = 1;
  sc.far = 120;
  sun.shadow.bias = -0.0005;
  sun.shadow.normalBias = 0.03;
  sun.shadow.radius = 3;
  sun.target.position.set(0, 0, -1);
  scene.add(sun, sun.target);
  const hemi = new THREE.HemisphereLight('#dfe6ee', '#b8a585', 0.35);
  scene.add(hemi);
  let envTex: THREE.Texture | null = null;

  // Cameras: one perspective, two orthographic (plan needs north-up)
  const persp = new THREE.PerspectiveCamera(38, 1, 0.1, 1500);
  const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, -200, 1500);
  const planCam = new THREE.OrthographicCamera(-1, 1, 1, -1, -200, 1500);
  planCam.up.set(0, 0, -1);
  const controls = new Map<THREE.Camera, OrbitControls>();
  for (const cam of [persp, ortho, planCam]) {
    cam.position.set(20, 20, 20);
    const c = new OrbitControls(cam, canvas);
    c.enableDamping = true;
    c.dampingFactor = 0.08;
    c.enabled = false;
    c.enableZoom = false;
    c.zoomSpeed = 0.8;
    c.screenSpacePanning = true;
    controls.set(cam, c);
  }
  canvas.style.touchAction = 'pan-y';
  let active: THREE.Camera = persp;
  let mode: ViewMode = 'eye';
  let orthoHeight = 24;
  let exploded = false;
  let stage: number = LAYERS.length;
  let section: Section = 'none';
  let dirty = true;
  const invalidate = () => {
    dirty = true;
  };
  const dur = (s: number) => (opts.reducedMotion ? 0 : s);

  const aspect = () => Math.max(0.2, container.clientWidth / Math.max(1, container.clientHeight));
  const orthoH = new Map<THREE.OrthographicCamera, number>([
    [ortho, 24],
    [planCam, 23],
  ]);
  const frameOrtho = (cam: THREE.OrthographicCamera, height: number) => {
    orthoH.set(cam, height);
    const a = aspect();
    // Keep the whole building in frame on narrow screens
    const h = height * Math.max(1, 1.2 / a);
    cam.top = h / 2;
    cam.bottom = -h / 2;
    cam.left = (-h / 2) * a;
    cam.right = (h / 2) * a;
    cam.updateProjectionMatrix();
  };
  const perspDistanceScale = () => {
    const a = aspect();
    return a < 1.2 ? Math.min(1.9, 1.25 / a) : 1;
  };

  // Gate wheel-zoom behind Ctrl/⌘ so normal scrolling still moves the page
  const gate = (e: WheelEvent) => {
    const c = controls.get(active);
    if (c) c.enableZoom = e.ctrlKey || e.metaKey;
  };
  container.addEventListener('wheel', gate, { capture: true, passive: true });
  for (const c of controls.values()) c.addEventListener('change', invalidate);

  // The exploded diagram uses a lower axonometric so the layer gaps read clearly
  const EXPLODED_VIEW: OrthoView = { kind: 'ortho', dir: [1, 0.6, 1.2], target: [0, 0, -3], height: 40, rotate: true, polar: [1.204, 1.204] };

  function applyView(v: ViewMode, animate: boolean) {
    const cfg = exploded ? EXPLODED_VIEW : VIEWS[v];
    const prev = active;
    const t = exploded ? new THREE.Vector3(0, ((LAYERS.length - 2) * EXPLODE_GAP + 7) / 2, -3) : new THREE.Vector3(...cfg.target);
    if (cfg.kind === 'persp') {
      active = persp;
      persp.fov = cfg.fov;
      persp.aspect = aspect();
      persp.updateProjectionMatrix();
      const c = controls.get(persp)!;
      c.minPolarAngle = cfg.polar[0];
      c.maxPolarAngle = cfg.polar[1];
      c.minDistance = cfg.dist[0];
      c.maxDistance = cfg.dist[1] * 1.6;
      c.enablePan = false;
      const pos = new THREE.Vector3(...cfg.pos).sub(new THREE.Vector3(...cfg.target)).multiplyScalar(perspDistanceScale()).add(t);
      if (animate && prev === persp) {
        gsap.to(persp.position, { x: pos.x, y: pos.y, z: pos.z, duration: dur(1.3), ease: 'power3.inOut', onUpdate: invalidate });
        gsap.to(c.target, { x: t.x, y: t.y, z: t.z, duration: dur(1.3), ease: 'power3.inOut', onUpdate: () => c.update() });
      } else {
        persp.position.copy(pos);
        c.target.copy(t);
      }
    } else {
      const cam = cfg.plan ? planCam : ortho;
      active = cam;
      const c = controls.get(cam)!;
      c.enableRotate = cfg.rotate;
      c.enablePan = true;
      if (cfg.polar) {
        c.minPolarAngle = cfg.polar[0];
        c.maxPolarAngle = cfg.polar[1];
      } else {
        c.minPolarAngle = 0;
        c.maxPolarAngle = Math.PI;
      }
      c.minZoom = 0.5;
      c.maxZoom = 6;
      orthoHeight = cfg.height;
      const pos = t.clone().addScaledVector(new THREE.Vector3(...cfg.dir).normalize(), 200);
      if (animate && prev === cam) {
        gsap.to(cam.position, { x: pos.x, y: pos.y, z: pos.z, duration: dur(1.2), ease: 'power3.inOut', onUpdate: invalidate });
        gsap.to(c.target, { x: t.x, y: t.y, z: t.z, duration: dur(1.2), ease: 'power3.inOut', onUpdate: () => c.update() });
        const box = { h: orthoH.get(cam) ?? orthoHeight, z: cam.zoom };
        gsap.to(box, {
          h: orthoHeight,
          z: 1,
          duration: dur(1.2),
          ease: 'power3.inOut',
          onUpdate: () => {
            cam.zoom = box.z;
            frameOrtho(cam, box.h);
            invalidate();
          },
        });
      } else {
        cam.position.copy(pos);
        c.target.copy(t);
        cam.zoom = 1;
        frameOrtho(cam, orthoHeight);
      }
    }
    groundPlane.constant = v === 'elevation' && !exploded ? 0.07 : 1e4;
    for (const [cam, c] of controls) c.enabled = cam === active;
    controls.get(active)!.update();
    if (prev !== active && !opts.reducedMotion) {
      canvas.style.opacity = '0.25';
      requestAnimationFrame(() => (canvas.style.opacity = '1'));
    }
    invalidate();
  }

  function layout(animate: boolean) {
    layers.forEach((l, i) => {
      const shown = i < stage && !(exploded && l.id === 'landscape');
      const baseY = exploded ? i * EXPLODE_GAP : 0;
      const g = l.group;
      gsap.killTweensOf(g.position);
      if (shown) {
        if (!g.visible) {
          g.visible = true;
          g.position.y = baseY + 9;
        }
        if (animate) gsap.to(g.position, { y: baseY, duration: dur(1.1), ease: 'power3.out', delay: dur(i * 0.04), onUpdate: invalidate });
        else g.position.y = baseY;
      } else if (g.visible) {
        if (animate && !opts.reducedMotion)
          gsap.to(g.position, {
            y: baseY + 9,
            duration: 0.6,
            ease: 'power2.in',
            onUpdate: invalidate,
            onComplete: () => {
              g.visible = false;
              invalidate();
            },
          });
        else g.visible = false;
      }
    });
    invalidate();
  }

  let envTimer: number | undefined;
  function setHour(h: number, settle = true) {
    const s = sunAt(h);
    sky.set(s.colours, s.dir);
    sun.color.set(s.lightColour);
    sun.intensity = s.intensity + 1.2;
    sun.position.copy(s.dir).multiplyScalar(60).add(sun.target.position);
    hemi.intensity = s.ambient;
    invalidate();
    window.clearTimeout(envTimer);
    const rebuild = () => {
      const next = sky.envMap(renderer);
      envTex?.dispose();
      envTex = next;
      scene.environment = envTex;
      scene.environmentIntensity = 0.42;
      invalidate();
    };
    if (settle) rebuild();
    else envTimer = window.setTimeout(rebuild, 180);
  }

  // Overlay labels follow their 3D anchors
  const v = new THREE.Vector3();
  const toCam = new THREE.Vector3();
  const facings = houseAnnotations.map((a) => new THREE.Vector3(...a.facing).normalize());
  const facesCamera = (i: number, anchor: THREE.Vector3) => {
    if ((active as THREE.OrthographicCamera).isOrthographicCamera) active.getWorldDirection(toCam).negate();
    else toCam.copy(active.position).sub(anchor).normalize();
    return facings[i].dot(toCam) > 0.08;
  };
  const place = (el: HTMLElement | null, world: THREE.Vector3, visible: boolean, clamp = false) => {
    if (!el) return;
    if (!visible) {
      el.style.opacity = '0';
      el.style.visibility = 'hidden';
      return;
    }
    v.copy(world).project(active);
    const inView = v.z < 1 && Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05;
    el.style.visibility = inView ? 'visible' : 'hidden';
    el.style.opacity = inView ? '1' : '0';
    let x = (v.x * 0.5 + 0.5) * container.clientWidth;
    // Keep side labels inside the frame on narrow screens
    if (clamp) x = Math.min(x, container.clientWidth - el.offsetWidth - 8);
    const y = (-v.y * 0.5 + 0.5) * container.clientHeight;
    el.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0)`;
  };
  function updateLabels() {
    const cut = CUT[section];
    ANNOTATIONS.forEach((a, i) => {
      const layer = layers[ANNOTATION_LAYER[i]];
      const ok = !exploded && layer.group.visible && a.anchor.y < cut && Math.abs(layer.group.position.y) < 0.2 && facesCamera(i, a.anchor);
      place(opts.annotationEls[i], a.anchor, ok);
    });
    layers.forEach((l, i) => {
      const box = l.box;
      const anchor = new THREE.Vector3(box.max.x + 0.4, (box.min.y + box.max.y) / 2 + l.group.position.y, box.max.z);
      if (l.id === 'furniture' || l.id === 'interiors') anchor.y = 1.5 + l.group.position.y;
      place(opts.layerEls[i], anchor, exploded && l.group.visible && l.id !== 'landscape', true);
    });
  }

  const resize = () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    persp.aspect = w / h;
    persp.updateProjectionMatrix();
    frameOrtho(ortho, orthoH.get(ortho) ?? 24);
    frameOrtho(planCam, orthoH.get(planCam) ?? 23);
    invalidate();
  };
  const ro = new ResizeObserver(resize);
  ro.observe(container);

  let running = false;
  const loop = () => {
    const moved = controls.get(active)!.update();
    if (moved || dirty) {
      dirty = false;
      renderer.render(scene, active);
      updateLabels();
    }
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

  resize();
  setHour(16);
  applyView('eye', false);
  layout(false);
  await renderer.compileAsync(scene, persp).catch(() => undefined);
  io.observe(container);
  loop();

  return {
    setView(next) {
      mode = next;
      applyView(next, true);
    },
    setStage(n) {
      stage = THREE.MathUtils.clamp(n, 1, LAYERS.length);
      layout(true);
    },
    setExploded(on) {
      exploded = on;
      if (on) {
        mode = 'axo';
        stage = LAYERS.length;
        section = 'none';
        cutPlane.constant = CUT.none;
      }
      layout(true);
      applyView(on ? 'axo' : mode, true);
    },
    setSection(s) {
      section = s;
      cutPlane.constant = CUT[s];
      invalidate();
    },
    setHour,
    zoom(f) {
      if (active === persp) {
        const c = controls.get(persp)!;
        const off = persp.position.clone().sub(c.target);
        off.setLength(THREE.MathUtils.clamp(off.length() * f, c.minDistance, c.maxDistance));
        gsap.to(persp.position, { x: c.target.x + off.x, y: c.target.y + off.y, z: c.target.z + off.z, duration: dur(0.5), ease: 'power2.out', onUpdate: invalidate });
      } else {
        const cam = active as THREE.OrthographicCamera;
        gsap.to(cam, {
          zoom: THREE.MathUtils.clamp(cam.zoom / f, 0.5, 6),
          duration: dur(0.5),
          ease: 'power2.out',
          onUpdate: () => {
            cam.updateProjectionMatrix();
            invalidate();
          },
        });
      }
    },
    reset() {
      applyView(exploded ? 'axo' : mode, true);
    },
    dispose() {
      window.clearTimeout(envTimer);
      io.disconnect();
      ro.disconnect();
      renderer.setAnimationLoop(null);
      container.removeEventListener('wheel', gate, { capture: true });
      layers.forEach((l) => gsap.killTweensOf(l.group.position));
      for (const c of controls.values()) c.dispose();
      root.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.isMesh) m.geometry.dispose();
      });
      poche.dispose();
      M.dispose();
      sky.dispose();
      envTex?.dispose();
      renderer.dispose();
      canvas.remove();
    },
  };
}
