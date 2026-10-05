'use client';

import { useCallback, useEffect, useRef, useState, type PointerEvent } from 'react';
import { SmartImage } from '@/components/ui/SmartImage';
import { Tilt } from '@/components/motion/Tilt';
import { products } from '@/lib/content';
import { cn } from '@/lib/cn';

/** Drag-to-scroll, snap-aligned product rail with prev/next controls. */
export function ProductCarousel() {
  const track = useRef<HTMLUListElement>(null);
  const drag = useRef<{ x: number; left: number; moved: boolean } | null>(null);
  const [edges, setEdges] = useState({ start: true, end: false });

  const updateEdges = useCallback(() => {
    const t = track.current;
    if (!t) return;
    setEdges({ start: t.scrollLeft < 8, end: t.scrollLeft + t.clientWidth > t.scrollWidth - 8 });
  }, []);

  useEffect(() => {
    updateEdges();
    window.addEventListener('resize', updateEdges);
    return () => window.removeEventListener('resize', updateEdges);
  }, [updateEdges]);

  const step = (dir: 1 | -1) => {
    const t = track.current;
    if (!t) return;
    const card = t.querySelector('li');
    const w = card ? card.getBoundingClientRect().width + 24 : t.clientWidth * 0.8;
    t.scrollBy({ left: dir * w, behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  };

  // Mouse drag (touch already scrolls natively)
  const onDown = (e: PointerEvent<HTMLUListElement>) => {
    if (e.pointerType !== 'mouse' || !track.current) return;
    drag.current = { x: e.clientX, left: track.current.scrollLeft, moved: false };
  };
  const onMove = (e: PointerEvent<HTMLUListElement>) => {
    const d = drag.current;
    const t = track.current;
    if (!d || !t) return;
    const dx = e.clientX - d.x;
    if (Math.abs(dx) > 4 && !d.moved) {
      d.moved = true;
      t.setPointerCapture(e.pointerId);
      t.style.scrollSnapType = 'none';
    }
    if (d.moved) t.scrollLeft = d.left - dx;
  };
  const onUp = () => {
    const t = track.current;
    if (t) t.style.scrollSnapType = '';
    setTimeout(() => (drag.current = null), 0);
  };

  return (
    <div>
      <div className="flex items-end justify-between gap-6">
        <h3 className="font-serif text-[clamp(1.9rem,3vw,2.6rem)] leading-none">From the workshop</h3>
        <div className="flex gap-2">
          <button type="button" onClick={() => step(-1)} disabled={edges.start} aria-label="Previous pieces" className="grid h-12 w-12 place-items-center rounded-full border border-indigo/25 transition-colors hover:bg-indigo hover:text-ivory disabled:opacity-35 disabled:hover:bg-transparent disabled:hover:text-indigo">
            <svg width="18" height="14" viewBox="0 0 18 14" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
              <path d="M17 7H2M7 1L1 7l6 6" />
            </svg>
          </button>
          <button type="button" onClick={() => step(1)} disabled={edges.end} aria-label="Next pieces" className="grid h-12 w-12 place-items-center rounded-full border border-indigo/25 transition-colors hover:bg-indigo hover:text-ivory disabled:opacity-35 disabled:hover:bg-transparent disabled:hover:text-indigo">
            <svg width="18" height="14" viewBox="0 0 18 14" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
              <path d="M1 7h15M11 1l6 6-6 6" />
            </svg>
          </button>
        </div>
      </div>
      <ul
        ref={track}
        data-cursor="Drag"
        aria-label="Furniture pieces (scrollable)"
        tabIndex={0}
        onScroll={updateEdges}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
        onClickCapture={(e) => {
          if (drag.current?.moved) e.preventDefault();
        }}
        className="mt-10 -mr-[clamp(1rem,4vw,3.5rem)] flex snap-x snap-mandatory gap-6 overflow-x-auto pr-[clamp(1rem,4vw,3.5rem)] pb-4 [scrollbar-width:none] select-none [&::-webkit-scrollbar]:hidden"
      >
        {products.map((p, i) => (
          <li key={p.name} className="w-[78%] shrink-0 snap-start xs:w-[58%] sm:w-[40%] lg:w-[26%]">
            <Tilt max={3}>
              <article className="group">
                <div className={cn('relative aspect-[4/5] overflow-hidden bg-[#e6dccc]', i % 2 ? 'rounded-t-[999px]' : 'arch-mughal')}>
                  <SmartImage src={p.image} alt={`${p.name}: ${p.materials}`} sizes="(min-width: 1024px) 24vw, (min-width: 640px) 40vw, 78vw" imgClassName="transition-transform duration-[1400ms] ease-out-expo group-hover:scale-[1.05]" />
                </div>
                <h4 className="mt-5 font-serif text-2xl">{p.name}</h4>
                <p className="mt-1 text-sm text-ink-muted">{p.materials}</p>
              </article>
            </Tilt>
          </li>
        ))}
      </ul>
    </div>
  );
}
