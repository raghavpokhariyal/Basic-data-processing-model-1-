'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { HouseViewerApi, Section, ViewMode } from '@/lib/three/houseViewer';
import { houseAnnotations, houseLayers, houseSections, houseViews } from '@/lib/house';
import { SplitReveal, Reveal } from '@/components/motion/Reveal';
import { SmartImage } from '@/components/ui/SmartImage';
import { prefersReducedMotion } from '@/lib/gsap';
import { img } from '@/lib/images';
import { cn } from '@/lib/cn';

const formatHour = (h: number) => {
  const hh = Math.floor(h);
  const mm = Math.round((h - hh) * 60);
  const suffix = hh >= 12 ? 'pm' : 'am';
  const h12 = hh % 12 === 0 ? 12 : hh % 12;
  return `${h12}:${String(mm).padStart(2, '0')} ${suffix}`;
};

function canUseWebGL() {
  try {
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') || c.getContext('webgl'));
  } catch {
    return false;
  }
}

function Segmented<T extends string>({
  label,
  options,
  value,
  onChange,
  disabled,
}: {
  label: string;
  options: readonly { id: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  disabled?: boolean;
}) {
  return (
    <div role="group" aria-label={label} className={cn('flex flex-wrap gap-1.5', disabled && 'opacity-45')}>
      {options.map((o) => (
        <button
          key={o.id}
          type="button"
          disabled={disabled}
          aria-pressed={value === o.id}
          onClick={() => onChange(o.id)}
          className={cn(
            'rounded-t-full rounded-b-md border px-3.5 py-2 text-[0.68rem] font-semibold tracking-[0.12em] uppercase transition-colors duration-300 disabled:cursor-not-allowed',
            value === o.id ? 'border-indigo bg-indigo text-ivory' : 'border-indigo/25 text-indigo hover:border-indigo',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function HouseExplorer() {
  const mount = useRef<HTMLDivElement>(null);
  const api = useRef<HouseViewerApi | null>(null);
  const pinRefs = useRef<(HTMLDivElement | null)[]>([]);
  const layerRefs = useRef<(HTMLDivElement | null)[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'unsupported'>('idle');
  const [progress, setProgress] = useState(0);
  const [view, setView] = useState<ViewMode>('eye');
  const [section, setSection] = useState<Section>('none');
  const [exploded, setExploded] = useState(false);
  const [stage, setStage] = useState<number>(houseLayers.length);
  const [hour, setHour] = useState(16);
  const [playing, setPlaying] = useState(false);
  const [activePin, setActivePin] = useState<number | null>(null);
  const autoSection = useRef(false);
  const playTimer = useRef<number | undefined>(undefined);

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
        // Let the loading state paint before the model is built
        await new Promise((r) => setTimeout(r, 60));
        const { createHouseViewer } = await import('@/lib/three/houseViewer');
        if (cancelled) return;
        try {
          const viewer = await createHouseViewer(el, {
            reducedMotion: prefersReducedMotion(),
            annotationEls: pinRefs.current,
            layerEls: layerRefs.current,
            onProgress: (p) => !cancelled && setProgress(p),
          });
          if (cancelled) {
            viewer.dispose();
            return;
          }
          api.current = viewer;
          setStatus('ready');
        } catch {
          if (!cancelled) setStatus('unsupported');
        }
      },
      { rootMargin: '600px 0px' },
    );
    io.observe(el);
    return () => {
      cancelled = true;
      io.disconnect();
      window.clearTimeout(playTimer.current);
      api.current?.dispose();
      api.current = null;
    };
  }, []);

  const stopPlay = useCallback(() => {
    window.clearTimeout(playTimer.current);
    setPlaying(false);
  }, []);

  const chooseView = (v: ViewMode) => {
    stopPlay();
    if (exploded) {
      setExploded(false);
      api.current?.setExploded(false);
    }
    setView(v);
    api.current?.setView(v);
    if (v === 'plan' && section === 'none') {
      autoSection.current = true;
      setSection('ground');
      api.current?.setSection('ground');
    } else if (v !== 'plan' && autoSection.current) {
      autoSection.current = false;
      setSection('none');
      api.current?.setSection('none');
    }
  };

  const chooseSection = (s: Section) => {
    autoSection.current = false;
    setSection(s);
    api.current?.setSection(s);
  };

  const toggleExplode = () => {
    stopPlay();
    const next = !exploded;
    setExploded(next);
    if (next) {
      setView('axo');
      setSection('none');
      setStage(houseLayers.length);
      autoSection.current = false;
    }
    api.current?.setExploded(next);
  };

  const chooseStage = (n: number) => {
    setStage(n);
    api.current?.setStage(n);
  };

  const play = () => {
    if (playing) return stopPlay();
    setPlaying(true);
    let n = 1;
    const tick = () => {
      chooseStage(n);
      if (n >= houseLayers.length) {
        setPlaying(false);
        return;
      }
      n += 1;
      playTimer.current = window.setTimeout(tick, prefersReducedMotion() ? 900 : 1500);
    };
    tick();
  };

  const currentLayer = houseLayers[stage - 1];
  const currentView = houseViews.find((v) => v.id === view)!;
  const ready = status === 'ready';
  const describe = `3D model of the villa in ${currentView.label.toLowerCase()} view, built to stage ${stage} of ${houseLayers.length}: ${currentLayer.title}${
    exploded ? ', pulled apart into an exploded diagram' : ''
  }${section !== 'none' ? `, cut through the ${section === 'ground' ? 'ground' : 'upper'} floor` : ''}.`;

  return (
    <section
      id="explore"
      aria-labelledby="explore-title"
      className="relative overflow-hidden bg-paper py-28 md:py-36"
      style={{
        backgroundImage:
          'linear-gradient(rgba(46,58,89,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(46,58,89,0.05) 1px, transparent 1px), linear-gradient(rgba(46,58,89,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(46,58,89,0.025) 1px, transparent 1px)',
        backgroundSize: '120px 120px, 120px 120px, 24px 24px, 24px 24px',
      }}
    >
      <div className="container-x">
        <div className="grid gap-10 lg:grid-cols-12 lg:items-end">
          <div className="lg:col-span-7">
            <Reveal>
              <p className="eyebrow flex items-center gap-3 text-terracotta-deep">
                <span className="h-px w-10 bg-current" aria-hidden="true" />
                The house in 3D
                <span lang="hi" className="text-[0.95rem] tracking-normal normal-case">
                  · त्रि-आयामी
                </span>
              </p>
            </Reveal>
            <SplitReveal as="h2" id="explore-title" className="display-lg mt-7 max-w-[15ch] font-serif">
              Walk around a home before it is built.
            </SplitReveal>
          </div>
          <Reveal className="lg:col-span-4 lg:col-start-9">
            <p className="text-lg leading-relaxed text-ink-muted">
              Every project starts as a model like this one. See it at eye level, read it in plan and elevation, watch it go up layer by layer, and move the sun to see how light falls through the jaali.
            </p>
          </Reveal>
        </div>

        {/* Controls */}
        <div className="mt-14 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="eyebrow mb-2 text-[0.62rem] text-ink-muted">View</p>
            <Segmented label="Camera view" options={houseViews} value={view} onChange={chooseView} />
          </div>
          <div className="flex flex-wrap items-end gap-6">
            <div>
              <p className="eyebrow mb-2 text-[0.62rem] text-ink-muted">Section cut</p>
              <Segmented label="Section cut" options={houseSections} value={section} onChange={chooseSection} disabled={exploded} />
            </div>
            <button
              type="button"
              aria-pressed={exploded}
              onClick={toggleExplode}
              className={cn(
                'rounded-t-full rounded-b-md border px-4 py-2 text-[0.68rem] font-semibold tracking-[0.12em] uppercase transition-colors duration-300',
                exploded ? 'border-terracotta-deep bg-terracotta-deep text-ivory' : 'border-terracotta-deep/50 text-terracotta-deep hover:border-terracotta-deep',
              )}
            >
              Exploded diagram
            </button>
          </div>
        </div>

        {/* Viewer */}
        <div className="relative mt-6 aspect-[4/5] overflow-hidden rounded-t-[min(16vw,9rem)] border border-brass/40 bg-[#e6e2d8] sm:aspect-[4/3] lg:aspect-[16/9]">
          <div className={cn('absolute inset-0 transition-opacity duration-700', ready ? 'pointer-events-none opacity-0' : 'opacity-100')} aria-hidden={ready}>
            <SmartImage src={img.heroVilla} alt="Rendered view of the villa" sizes="(min-width: 1024px) 90vw, 100vw" />
            <div className="absolute inset-0 bg-paper/40" />
            {(status === 'loading' || status === 'idle') && (
              <div className="absolute inset-x-0 bottom-8 mx-auto w-64 text-center" aria-live="polite">
                <p className="eyebrow text-[0.65rem] text-indigo">Building the model · {Math.round(progress * 100)}%</p>
                <div className="mt-3 h-px bg-indigo/15">
                  <div className="h-px origin-left bg-terracotta-deep transition-transform duration-300" style={{ transform: `scaleX(${progress})` }} />
                </div>
              </div>
            )}
            {status === 'unsupported' && <p className="absolute inset-x-6 bottom-8 text-center text-sm text-indigo">The 3D model needs WebGL, which this browser has turned off.</p>}
          </div>

          <div ref={mount} data-cursor="Orbit" role="img" aria-label={describe} className="absolute inset-0">
            {houseAnnotations.map((a, i) => (
              <div
                key={a.n}
                ref={(el) => {
                  pinRefs.current[i] = el;
                }}
                aria-hidden="true"
                className="pointer-events-none absolute top-0 left-0 opacity-0 transition-opacity duration-300"
              >
                <div className="flex -translate-x-[0.9rem] -translate-y-1/2 items-center gap-2">
                  <span
                    className={cn(
                      'grid h-7 w-7 shrink-0 place-items-center rounded-full border-2 border-ivory text-[0.7rem] font-bold text-ivory shadow-[0_6px_16px_-6px_rgba(29,37,57,0.7)] transition-transform duration-300',
                      activePin === a.n ? 'scale-125 bg-indigo' : 'bg-terracotta-deep',
                    )}
                  >
                    {a.n}
                  </span>
                  <span className="hidden rounded-full bg-ivory/92 px-3 py-1 text-[0.7rem] font-semibold whitespace-nowrap text-indigo shadow-sm md:inline">{a.title}</span>
                </div>
              </div>
            ))}
            {houseLayers.map((l, i) => (
              <div
                key={l.id}
                ref={(el) => {
                  layerRefs.current[i] = el;
                }}
                aria-hidden="true"
                className="pointer-events-none absolute top-0 left-0 opacity-0 transition-opacity duration-300"
              >
                <div className="flex -translate-y-1/2 items-center gap-2">
                  <span className="h-px w-6 bg-indigo/50" />
                  <span className="rounded-full bg-indigo px-3 py-1 text-[0.68rem] font-semibold whitespace-nowrap text-ivory">
                    0{i + 1} {l.title}
                    <span lang="hi" className="ml-2 font-deva text-sand">
                      {l.hindi}
                    </span>
                  </span>
                </div>
              </div>
            ))}
          </div>

          {ready && (
            <>
              <p className="pointer-events-none absolute top-4 left-1/2 hidden -translate-x-1/2 rounded-full bg-ivory/85 px-3 py-1 text-[0.68rem] font-semibold tracking-[0.08em] text-indigo sm:block">
                {exploded ? 'Exploded axonometric' : currentView.label}
                {section !== 'none' && !exploded ? ` · cut ${section === 'ground' ? 'ground' : 'upper'} floor` : ''}
              </p>
              <div className="absolute bottom-3 left-3 flex gap-1.5 sm:right-4 sm:bottom-4 sm:left-auto">
                {[
                  { label: 'Zoom in', on: () => api.current?.zoom(0.8), d: 'M8 3v10M3 8h10' },
                  { label: 'Zoom out', on: () => api.current?.zoom(1.25), d: 'M3 8h10' },
                  { label: 'Reset view', on: () => api.current?.reset(), d: 'M3 8a5 5 0 1 0 1.5-3.5M3 2.5V5h2.5' },
                ].map((b) => (
                  <button
                    key={b.label}
                    type="button"
                    onClick={b.on}
                    aria-label={b.label}
                    title={b.label}
                    className="grid h-10 w-10 place-items-center rounded-full bg-ivory/92 text-indigo shadow-sm transition-colors hover:bg-indigo hover:text-ivory"
                  >
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
                      <path d={b.d} />
                    </svg>
                  </button>
                ))}
              </div>
              <p className="pointer-events-none absolute bottom-4 left-4 hidden rounded-full bg-ivory/85 px-3 py-1 text-[0.68rem] text-indigo sm:block">
                Drag to orbit · Ctrl/⌘ + scroll to zoom
              </p>
            </>
          )}
        </div>

        {/* Build sequence + sunlight */}
        <div className="mt-8 grid gap-10 lg:grid-cols-12">
          <div className="lg:col-span-8">
            <div className="flex items-center justify-between gap-4">
              <p className="eyebrow text-[0.62rem] text-ink-muted">Build sequence</p>
              <button
                type="button"
                onClick={play}
                disabled={!ready}
                className="flex items-center gap-2 text-[0.7rem] font-semibold tracking-[0.12em] text-terracotta-deep uppercase disabled:opacity-40"
              >
                <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" aria-hidden="true">
                  {playing ? <path d="M2 1h3v10H2zM7 1h3v10H7z" /> : <path d="M2 1l9 5-9 5z" />}
                </svg>
                {playing ? 'Pause' : 'Play the build'}
              </button>
            </div>
            <ol className="mt-3 grid grid-cols-7 gap-1.5" aria-label="Construction stages">
              {houseLayers.map((l, i) => (
                <li key={l.id}>
                  <button
                    type="button"
                    disabled={!ready}
                    onClick={() => {
                      stopPlay();
                      chooseStage(i + 1);
                    }}
                    aria-current={stage === i + 1 ? 'step' : undefined}
                    aria-label={`Stage ${i + 1}: ${l.title}`}
                    className="group w-full text-left disabled:cursor-not-allowed"
                  >
                    <span className={cn('block h-1.5 rounded-full transition-colors duration-500', i < stage ? 'bg-terracotta-deep' : 'bg-indigo/15 group-hover:bg-indigo/30')} />
                    <span className={cn('mt-2 hidden text-[0.68rem] leading-tight font-semibold sm:block', i < stage ? 'text-indigo' : 'text-ink-muted')}>{l.title}</span>
                  </button>
                </li>
              ))}
            </ol>
            <div className="mt-5 flex gap-4" aria-live="polite">
              <p className="font-serif text-4xl leading-none text-terracotta-deep tabular-nums">0{stage}</p>
              <div>
                <p className="font-serif text-2xl leading-tight">
                  {currentLayer.title}
                  <span lang="hi" className="ml-3 font-deva text-lg text-terracotta">
                    {currentLayer.hindi}
                  </span>
                </p>
                <p className="mt-1 max-w-xl text-ink-muted">{currentLayer.text}</p>
              </div>
            </div>
          </div>
          <div className="lg:col-span-4">
            <label htmlFor="house-sun" className="eyebrow flex items-center justify-between text-[0.62rem] text-ink-muted">
              <span>Sunlight</span>
              <span className="text-[0.8rem] tracking-normal text-indigo normal-case tabular-nums">{formatHour(hour)}</span>
            </label>
            <input
              id="house-sun"
              type="range"
              min={6}
              max={18.5}
              step={0.25}
              value={hour}
              disabled={!ready}
              onChange={(e) => {
                const h = Number(e.target.value);
                setHour(h);
                api.current?.setHour(h, false);
              }}
              aria-valuetext={formatHour(hour)}
              className="mt-4 w-full accent-terracotta-deep disabled:opacity-40"
            />
            <div className="mt-1 flex justify-between text-[0.65rem] text-ink-muted" aria-hidden="true">
              <span>Sunrise · east</span>
              <span>Noon · south</span>
              <span>Sunset · west</span>
            </div>
            <p className="mt-6 text-sm leading-relaxed text-ink-muted">
              <span className="font-semibold text-indigo">{exploded ? 'Exploded axonometric' : currentView.label}.</span>{' '}
              {exploded ? 'Each construction layer lifted apart and stacked in the order it is built.' : currentView.text}
            </p>
          </div>
        </div>

        {/* Legend */}
        <ol className="mt-14 grid gap-x-10 gap-y-6 border-t border-brass/30 pt-10 sm:grid-cols-2 lg:grid-cols-3">
          {houseAnnotations.map((a) => (
            <li key={a.n} className="flex gap-4" onMouseEnter={() => setActivePin(a.n)} onMouseLeave={() => setActivePin(null)}>
              <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-terracotta-deep text-[0.7rem] font-bold text-ivory" aria-hidden="true">
                {a.n}
              </span>
              <div>
                <p className="font-serif text-xl leading-tight">{a.title}</p>
                <p className="mt-1 text-sm leading-relaxed text-ink-muted">{a.text}</p>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
