import * as THREE from 'three';

export interface SkyColours {
  zenith: string;
  horizon: string;
  ground: string;
  sun: string;
}

/** Gradient sky dome with a soft sun glow; also used to light the scene. */
export function createSky(radius = 400) {
  const uniforms = {
    zenith: { value: new THREE.Color('#9fb0c0') },
    horizon: { value: new THREE.Color('#f1dcc0') },
    ground: { value: new THREE.Color('#cdbb9f') },
    sunDir: { value: new THREE.Vector3(1, 0.3, 0.5).normalize() },
    sunColor: { value: new THREE.Color('#ffd8a8') },
    glow: { value: 0.6 },
  };
  const material = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms,
    vertexShader: 'varying vec3 vDir; void main(){ vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
    fragmentShader: `
      uniform vec3 zenith; uniform vec3 horizon; uniform vec3 ground; uniform vec3 sunDir; uniform vec3 sunColor; uniform float glow;
      varying vec3 vDir;
      void main(){
        float h = vDir.y;
        vec3 col = h > 0.0 ? mix(horizon, zenith, pow(clamp(h,0.0,1.0), 0.55)) : mix(horizon, ground, clamp(-h*4.0,0.0,1.0));
        float s = max(dot(normalize(vDir), sunDir), 0.0);
        col += sunColor * (pow(s, 8.0) * 0.35 + pow(s, 64.0) * 0.6) * glow;
        gl_FragColor = vec4(col, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 48, 24), material);
  mesh.frustumCulled = false;
  mesh.renderOrder = -1;

  return {
    mesh,
    set(c: SkyColours, sunDir: THREE.Vector3, glow = 0.6) {
      uniforms.zenith.value.set(c.zenith);
      uniforms.horizon.value.set(c.horizon);
      uniforms.ground.value.set(c.ground);
      uniforms.sunColor.value.set(c.sun);
      uniforms.sunDir.value.copy(sunDir).normalize();
      uniforms.glow.value = glow;
    },
    /** Pre-filtered environment map from the current sky, for soft image-based light. */
    envMap(renderer: THREE.WebGLRenderer) {
      const s = new THREE.Scene();
      const m = new THREE.Mesh(mesh.geometry, material);
      s.add(m);
      const pm = new THREE.PMREMGenerator(renderer);
      const t = pm.fromScene(s, 0.04, 0.1, radius * 3).texture;
      pm.dispose();
      return t;
    },
    dispose() {
      mesh.geometry.dispose();
      material.dispose();
    },
  };
}

const mixHex = (a: string, b: string, t: number) => '#' + new THREE.Color(a).lerp(new THREE.Color(b), t).getHexString();

/**
 * Sun direction and sky palette for an hour of the day (6:00–18:30) on
 * a site in north India, east = +X, south = +Z (the front of the house).
 */
export function sunAt(hour: number) {
  const t = THREE.MathUtils.clamp((hour - 6) / 12.5, 0, 1); // 0 sunrise … 1 sunset
  const azimuth = THREE.MathUtils.lerp(-Math.PI * 0.5, Math.PI * 0.5, t); // east → south → west
  const elevation = Math.sin(t * Math.PI) * THREE.MathUtils.degToRad(68) + THREE.MathUtils.degToRad(4);
  const dir = new THREE.Vector3(Math.cos(elevation) * -Math.sin(azimuth), Math.sin(elevation), Math.cos(elevation) * Math.cos(azimuth)).normalize();
  const low = 1 - Math.min(1, Math.sin(t * Math.PI) * 1.6); // 1 at the horizon, 0 near noon
  const colours: SkyColours = {
    zenith: mixHex('#a9bccd', '#8e9cb2', low * 0.6),
    horizon: mixHex('#e9e6dc', '#f0d2b0', low),
    ground: mixHex('#cfc4b0', '#c4ac8c', low),
    sun: mixHex('#fff3df', '#ffbf80', low),
  };
  return {
    dir,
    colours,
    lightColour: mixHex('#fff4e6', '#ffc58f', low),
    intensity: 2.2 + Math.sin(t * Math.PI) * 1.6,
    ambient: 0.35 + Math.sin(t * Math.PI) * 0.25,
  };
}
