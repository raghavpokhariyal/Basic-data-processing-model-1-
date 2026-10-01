'use client';

import { useCallback, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import { gsap, useGSAP, MQ } from '@/lib/gsap';
import { SmartImage } from '@/components/ui/SmartImage';
import { beforeAfter } from '@/lib/content';

/** Drag (or use arrow keys) to compare the bare shell with the finished room. */
export function BeforeAfter() {
  const box = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState(50);
  const dragging = useRef(false);

  const fromClientX = useCallback((x: number) => {
    const b = box.current?.getBoundingClientRect();
    if (!b) return;
    setPos(Math.min(100, Math.max(0, ((x - b.left) / b.width) * 100)));
  }, []);

  // A gentle sweep the first time it comes into view hints that it moves
  useGSAP(
    () => {
      const mm = gsap.matchMedia();
      mm.add(MQ.motion, () => {
        const o = { v: 50 };
        gsap
          .timeline({ scrollTrigger: { trigger: box.current, start: 'top 70%', once: true } })
          .to(o, { v: 74, duration: 1.1, ease: 'power2.inOut', onUpdate: () => !dragging.current && setPos(o.v) })
          .to(o, { v: 50, duration: 1.3, ease: 'power3.inOut', onUpdate: () => !dragging.current && setPos(o.v) });
      });
    },
    { scope: box },
  );

  const onDown = (e: PointerEvent<HTMLDivElement>) => {
    dragging.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    fromClientX(e.clientX);
  };
  const onMove = (e: PointerEvent<HTMLDivElement>) => {
    if (dragging.current) fromClientX(e.clientX);
  };
  const onUp = () => {
    dragging.current = false;
  };
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 10 : 4;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') setPos((p) => Math.max(0, p - step));
    else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') setPos((p) => Math.min(100, p + step));
    else if (e.key === 'Home') setPos(0);
    else if (e.key === 'End') setPos(100);
    else return;
    e.preventDefault();
  };

  return (
    <div className="grid gap-10 lg:grid-cols-12 lg:items-center">
      <div className="lg:col-span-4">
        <p className="eyebrow text-terracotta-deep">Before &amp; after</p>
        <h3 className="mt-5 font-serif text-[clamp(2.2rem,3.6vw,3.25rem)] leading-[1.02]">{beforeAfter.title}</h3>
        <p className="eyebrow mt-3 text-ink-muted">{beforeAfter.place} · Interiors</p>
        <p className="mt-6 max-w-sm text-lg leading-relaxed text-ink-muted">{beforeAfter.text}</p>
        <p className="mt-6 text-sm text-ink-muted">Drag the handle, or focus it and use the arrow keys.</p>
      </div>
      <div className="lg:col-span-8">
        <div
          ref={box}
          data-cursor="Drag"
          className="relative aspect-[3/2] touch-pan-y overflow-hidden rounded-t-[min(14vw,10rem)] bg-sand select-none"
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
        >
          <SmartImage src={beforeAfter.after.src} alt={beforeAfter.after.alt} sizes="(min-width: 1024px) 62vw, 94vw" />
          <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
            <SmartImage src={beforeAfter.before.src} alt={beforeAfter.before.alt} sizes="(min-width: 1024px) 62vw, 94vw" />
          </div>
          <span className="eyebrow pointer-events-none absolute bottom-5 left-5 bg-indigo-deep/80 px-3 py-1.5 text-[0.62rem] text-ivory">Before</span>
          <span className="eyebrow pointer-events-none absolute right-5 bottom-5 bg-ivory/90 px-3 py-1.5 text-[0.62rem] text-indigo">After</span>
          <div className="pointer-events-none absolute inset-y-0 w-px bg-ivory/90" style={{ left: `${pos}%` }} />
          <div
            role="slider"
            tabIndex={0}
            aria-label="Before and after comparison: share of the before image shown"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(pos)}
            aria-valuetext={`${Math.round(pos)}% before`}
            onKeyDown={onKey}
            className="absolute top-1/2 grid h-16 w-12 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize place-items-center rounded-t-full rounded-b-lg bg-ivory text-indigo shadow-[0_10px_30px_-10px_rgba(29,37,57,0.6)]"
            style={{ left: `${pos}%` }}
          >
            <svg width="22" height="12" viewBox="0 0 22 12" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
              <path d="M6 1L1 6l5 5M16 1l5 5-5 5" />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}
