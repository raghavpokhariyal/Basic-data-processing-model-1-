'use client';

import { useEffect, useRef, useState } from 'react';
import type { PieceId, ViewerApi } from '@/lib/three/viewer';
import type { WoodSpecies } from '@/lib/three/textures';
import { SmartImage } from '@/components/ui/SmartImage';
import { ArchButton } from '@/components/ui/ArchButton';
import { prefersReducedMotion } from '@/lib/gsap';
import { furniturePieces, type FurniturePieceId } from '@/lib/content';
import { cn } from '@/lib/cn';

const finishes: { id: WoodSpecies; label: string; swatch: string }[] = [
  { id: 'sheesham', label: 'Sheesham', swatch: '#6e4228' },
  { id: 'teak', label: 'Teak', swatch: '#a4723f' },
  { id: 'mango', label: 'Mango wood', swatch: '#c09a6c' },
];

/** Default timber per piece, mirrored from the 3D models. */
const NATIVE: Record<FurniturePieceId, WoodSpecies | null> = {
  lounge: 'sheesham',
  dining: 'sheesham',
  charpai: 'sheesham',
  chowki: 'mango',
  console: 'teak',
  sideboard: 'mango',
  jhoola: 'sheesham',
  mudha: null,
};

function canUseWebGL() {
  try {
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') || c.getContext('webgl'));
  } catch {
    return false;
  }
}

/**
 * The workshop in 3D: every piece is a live model. Switch pieces and
 * timber, rotate and zoom, or pull a piece apart to see how it is made.
 * three.js loads only when the viewer approaches the viewport.
 */
export function ChairViewer() {
  const mount = useRef<HTMLDivElement>(null);
  const api = useRef<ViewerApi | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'unsupported'>('idle');
  const [pieceId, setPieceId] = useState<FurniturePieceId>('lounge');
  const [species, setSpecies] = useState<WoodSpecies | null>('sheesham');
  const [exploded, setExploded] = useState(false);
  const piece = furniturePieces.find((p) => p.id === pieceId)!;
  const ready = status === 'ready';

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
          piece: 'lounge',
          species: 'sheesham',
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

  const choosePiece = (id: FurniturePieceId) => {
    setPieceId(id);
    setSpecies(NATIVE[id]);
    api.current?.setPiece(id as PieceId);
  };
  const chooseSpecies = (s: WoodSpecies) => {
    setSpecies(s);
    api.current?.setSpecies(s);
  };
  const toggleExplode = () => {
    const next = !exploded;
    setExploded(next);
    api.current?.setExploded(next);
  };

  return (
    <div className="grid gap-14 lg:grid-cols-12 lg:items-start">
      <div className="lg:col-span-7">
        <div role="group" aria-label="Choose a piece" className="-mx-1 mb-4 flex gap-1.5 overflow-x-auto px-1 pb-2 [scrollbar-width:thin]">
          {furniturePieces.map((p) => (
            <button
              key={p.id}
              type="button"
              aria-pressed={pieceId === p.id}
              onClick={() => choosePiece(p.id)}
              className={cn(
                'shrink-0 rounded-t-full rounded-b-md border px-3.5 py-2 text-[0.68rem] font-semibold tracking-[0.1em] whitespace-nowrap uppercase transition-colors duration-300',
                pieceId === p.id ? 'border-indigo bg-indigo text-ivory' : 'border-indigo/25 text-indigo hover:border-indigo',
              )}
            >
              {p.name}
            </button>
          ))}
        </div>

        <div className="relative aspect-[4/5] overflow-hidden rounded-t-[999px] bg-[radial-gradient(120%_80%_at_50%_30%,#fbf8f2_0%,#efe6d6_55%,#e2d3bc_100%)] sm:aspect-square lg:aspect-[5/5.4]">
          <div aria-hidden="true" className="arch absolute inset-[6%] border border-brass/40" />
          <div className={cn('absolute inset-0 transition-opacity duration-700', ready ? 'opacity-0' : 'opacity-100')}>
            <SmartImage src={piece.poster} alt="" sizes="(min-width: 1024px) 50vw, 92vw" imgClassName="mix-blend-multiply" />
          </div>
          <div
            ref={mount}
            data-cursor="Rotate"
            role="img"
            aria-label={`Interactive 3D model of the ${piece.name}${exploded ? ', pulled apart to show its parts' : ''}. Drag to rotate; use the buttons below to rotate and zoom.`}
            className={cn('absolute inset-0 transition-opacity duration-700 [&>canvas]:block', ready ? 'opacity-100' : 'opacity-0')}
          />
          {status === 'loading' && (
            <p className="eyebrow absolute inset-x-0 bottom-6 text-center text-ink-muted" aria-live="polite">
              Loading 3D model…
            </p>
          )}
          {status === 'unsupported' && <p className="absolute inset-x-6 bottom-6 text-center text-sm text-ink-muted">3D preview is not available on this device.</p>}
        </div>

        <div className="mt-6 flex flex-wrap items-center justify-between gap-4">
          <div role="group" aria-label="Wood finish" className={cn('flex flex-wrap gap-2', !species && 'opacity-40')}>
            {finishes.map((f) => (
              <button
                key={f.id}
                type="button"
                disabled={!species}
                aria-pressed={species === f.id}
                onClick={() => chooseSpecies(f.id)}
                className={cn(
                  'flex items-center gap-2 rounded-full border py-2 pr-4 pl-2 text-sm transition-colors duration-300 disabled:cursor-not-allowed',
                  species === f.id ? 'border-indigo bg-indigo text-ivory' : 'border-indigo/25 hover:border-indigo',
                )}
              >
                <span className="h-5 w-5 rounded-full ring-1 ring-black/10" style={{ background: f.swatch }} aria-hidden="true" />
                {f.label}
              </button>
            ))}
          </div>
          <div role="group" aria-label="View controls" className="flex gap-1.5">
            <button
              type="button"
              aria-pressed={exploded}
              disabled={!ready}
              onClick={toggleExplode}
              className={cn(
                'mr-1 rounded-full border px-4 text-[0.68rem] font-semibold tracking-[0.12em] uppercase transition-colors disabled:opacity-40',
                exploded ? 'border-terracotta-deep bg-terracotta-deep text-ivory' : 'border-terracotta-deep/50 text-terracotta-deep hover:border-terracotta-deep',
              )}
            >
              {exploded ? 'Assemble' : 'Explode'}
            </button>
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
        <p className="mt-3 text-xs text-ink-muted">Drag to rotate · pinch or Ctrl/⌘ + scroll to zoom · Explode to see every part</p>
      </div>

      <div className="lg:col-span-4 lg:col-start-9 lg:pt-16" aria-live="polite">
        <p className="eyebrow text-terracotta-deep">From the workshop · in 3D</p>
        <h3 className="mt-4 font-serif text-[clamp(2.4rem,4vw,3.6rem)] leading-none">{piece.name}</h3>
        <p lang="hi" aria-hidden="true" className="mt-2 text-2xl text-terracotta">
          {piece.hindi}
        </p>
        <p className="mt-6 text-lg leading-relaxed text-ink-muted">{piece.text}</p>
        <dl className="mt-8 divide-y divide-brass/30 border-y border-brass/30">
          {[
            { label: 'Dimensions', value: piece.dims },
            { label: 'Timber', value: species ? finishes.find((f) => f.id === species)!.label : 'Reed, jute and cotton' },
            { label: 'Lead time', value: 'Made to order · 6–8 weeks' },
          ].map((s) => (
            <div key={s.label} className="flex justify-between gap-6 py-3 text-sm">
              <dt className="text-ink-muted">{s.label}</dt>
              <dd className="text-right font-semibold">{s.value}</dd>
            </div>
          ))}
        </dl>
        <div className="mt-10">
          <ArchButton href="/#contact">Commission a piece</ArchButton>
        </div>
      </div>
    </div>
  );
}
