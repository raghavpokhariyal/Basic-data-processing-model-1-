'use client';

import { useEffect, useRef, useState } from 'react';
import type { ViewerApi } from '@/lib/three/viewer';
import type { WoodSpecies } from '@/lib/three/textures';
import { SmartImage } from '@/components/ui/SmartImage';
import { prefersReducedMotion } from '@/lib/gsap';
import { img } from '@/lib/images';
import { cn } from '@/lib/cn';

const finishes: { id: WoodSpecies; label: string; swatch: string }[] = [
  { id: 'sheesham', label: 'Sheesham', swatch: '#6e4228' },
  { id: 'teak', label: 'Teak', swatch: '#a4723f' },
  { id: 'mango', label: 'Mango wood', swatch: '#c09a6c' },
];

function canUseWebGL() {
  try {
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') || c.getContext('webgl'));
  } catch {
    return false;
  }
}

/**
 * Loads three.js only when the viewer approaches the viewport. A rendered
 * poster stands in until the model is ready (or if WebGL is unavailable).
 */
export function ChairViewer() {
  const mount = useRef<HTMLDivElement>(null);
  const api = useRef<ViewerApi | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'unsupported'>('idle');
  const [species, setSpecies] = useState<WoodSpecies>('sheesham');
  const speciesRef = useRef(species);

  useEffect(() => {
    const el = mount.current;
    if (!el) return;
    let cancelled = false;
    const io = new IntersectionObserver(
      async ([entry]) => {
        if (!entry.isIntersecting) return;
        io.disconnect();
        if (!canUseWebGL()) {
          setStatus('unsupported');
          return;
        }
        setStatus('loading');
        const { createViewer } = await import('@/lib/three/viewer');
        if (cancelled) return;
        api.current = createViewer(el, {
          species: speciesRef.current,
          reducedMotion: prefersReducedMotion(),
          onReady: () => setStatus('ready'),
        });
      },
      { rootMargin: '400px 0px' },
    );
    io.observe(el);
    return () => {
      cancelled = true;
      io.disconnect();
      api.current?.dispose();
      api.current = null;
    };
  }, []);

  useEffect(() => {
    speciesRef.current = species;
    api.current?.setSpecies(species);
  }, [species]);

  const ready = status === 'ready';

  return (
    <div>
      <div className="relative aspect-[4/5] overflow-hidden rounded-t-[999px] bg-[radial-gradient(120%_80%_at_50%_30%,#fbf8f2_0%,#efe6d6_55%,#e2d3bc_100%)] sm:aspect-square lg:aspect-[5/5.4]">
        <div aria-hidden="true" className="arch absolute inset-[6%] border border-brass/40" />
        <div className={cn('absolute inset-0 transition-opacity duration-700', ready ? 'opacity-0' : 'opacity-100')}>
          <SmartImage src={img.serviceFurniture} alt="" sizes="(min-width: 1024px) 50vw, 92vw" imgClassName="mix-blend-multiply" />
        </div>
        <div
          ref={mount}
          data-cursor="Rotate"
          role="img"
          aria-label="Interactive 3D model of the Jharokha lounge chair. Drag to rotate; use the buttons below to rotate and zoom."
          className={cn('absolute inset-0 transition-opacity duration-700 [&>canvas]:block', ready ? 'opacity-100' : 'opacity-0')}
        />
        {status === 'loading' && (
          <p className="eyebrow absolute inset-x-0 bottom-6 text-center text-ink-muted" aria-live="polite">
            Loading 3D model…
          </p>
        )}
        {status === 'unsupported' && (
          <p className="absolute inset-x-6 bottom-6 text-center text-sm text-ink-muted">3D preview is not available on this device.</p>
        )}
      </div>

      <div className="mt-6 flex flex-wrap items-center justify-between gap-4">
        <div role="group" aria-label="Wood finish" className="flex flex-wrap gap-2">
          {finishes.map((f) => (
            <button
              key={f.id}
              type="button"
              aria-pressed={species === f.id}
              onClick={() => setSpecies(f.id)}
              className={cn(
                'flex items-center gap-2 rounded-full border py-2 pr-4 pl-2 text-sm transition-colors duration-300',
                species === f.id ? 'border-indigo bg-indigo text-ivory' : 'border-indigo/25 hover:border-indigo',
              )}
            >
              <span className="h-5 w-5 rounded-full ring-1 ring-black/10" style={{ background: f.swatch }} aria-hidden="true" />
              {f.label}
            </button>
          ))}
        </div>
        <div role="group" aria-label="View controls" className="flex gap-1.5">
          {[
            { label: 'Rotate left', on: () => api.current?.rotate(-Math.PI / 6), d: 'M8 3L3 8l5 5M3 8h10' },
            { label: 'Rotate right', on: () => api.current?.rotate(Math.PI / 6), d: 'M8 3l5 5-5 5M13 8H3' },
            { label: 'Zoom in', on: () => api.current?.zoom(0.85), d: 'M8 3v10M3 8h10' },
            { label: 'Zoom out', on: () => api.current?.zoom(1.18), d: 'M3 8h10' },
            { label: 'Reset view', on: () => api.current?.reset(), d: 'M3 8a5 5 0 1 0 1.5-3.5M3 2.5V5h2.5' },
          ].map((b) => (
            <button
              key={b.label}
              type="button"
              onClick={b.on}
              disabled={!ready}
              aria-label={b.label}
              title={b.label}
              className="grid h-10 w-10 place-items-center rounded-full border border-indigo/25 transition-colors hover:border-indigo hover:bg-indigo hover:text-ivory disabled:opacity-40"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
                <path d={b.d} />
              </svg>
            </button>
          ))}
        </div>
      </div>
      <p className="mt-3 text-xs text-ink-muted">Drag to rotate · pinch or Ctrl/⌘ + scroll to zoom</p>
    </div>
  );
}
