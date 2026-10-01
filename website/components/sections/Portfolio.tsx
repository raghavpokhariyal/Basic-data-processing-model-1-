'use client';

import { useMemo, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { gsap, Flip, ScrollTrigger, prefersReducedMotion } from '@/lib/gsap';
import { SplitReveal, Reveal } from '@/components/motion/Reveal';
import { SmartImage } from '@/components/ui/SmartImage';
import { StarRule } from '@/components/ornaments/Jaali';
import { Lightbox } from './Lightbox';
import { BeforeAfter } from './BeforeAfter';
import { categories, projects, type Category } from '@/lib/content';
import { cn } from '@/lib/cn';

type Filter = 'All' | Category;

export function Portfolio() {
  const [filter, setFilter] = useState<Filter>('All');
  const [open, setOpen] = useState<number | null>(null);
  const grid = useRef<HTMLUListElement>(null);
  const visible = useMemo(() => projects.filter((p) => filter === 'All' || p.category === filter), [filter]);

  const choose = (next: Filter) => {
    if (next === filter) return;
    const items = grid.current?.querySelectorAll('[data-item]');
    if (!items || prefersReducedMotion()) {
      setFilter(next);
      requestAnimationFrame(() => ScrollTrigger.refresh());
      return;
    }
    const state = Flip.getState(items);
    flushSync(() => setFilter(next));
    Flip.from(state, {
      duration: 0.85,
      ease: 'expo.inOut',
      absolute: true,
      stagger: 0.025,
      onEnter: (els) => gsap.fromTo(els, { opacity: 0, scale: 0.92 }, { opacity: 1, scale: 1, duration: 0.8, ease: 'expo.out', delay: 0.25 }),
      onLeave: (els) => gsap.to(els, { opacity: 0, scale: 0.92, duration: 0.45, ease: 'power2.in' }),
      onComplete: () => ScrollTrigger.refresh(),
    });
  };

  return (
    <section id="work" aria-labelledby="work-title" className="relative bg-ivory py-28 md:py-40">
      <div className="container-x">
        <div className="flex flex-col gap-10 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <Reveal>
              <p className="eyebrow flex items-center gap-3 text-terracotta-deep">
                <span className="h-px w-10 bg-current" aria-hidden="true" />
                Selected work
                <span lang="hi" className="text-[0.95rem] tracking-normal normal-case">
                  · हमारा काम
                </span>
              </p>
            </Reveal>
            <SplitReveal as="h2" id="work-title" className="display-lg mt-7 max-w-[13ch] font-serif">
              Homes, rooms and things made by hand.
            </SplitReveal>
          </div>
          <div role="group" aria-label="Filter projects by category" className="flex flex-wrap gap-2">
            {categories.map((c) => (
              <button
                key={c}
                type="button"
                aria-pressed={filter === c}
                onClick={() => choose(c)}
                className={cn(
                  'rounded-t-full rounded-b-md border px-5 py-2.5 text-[0.72rem] font-semibold tracking-[0.14em] uppercase transition-colors duration-500',
                  filter === c ? 'border-indigo bg-indigo text-ivory' : 'border-indigo/25 text-indigo hover:border-indigo',
                )}
              >
                {c}
              </button>
            ))}
          </div>
        </div>
        <p className="sr-only" aria-live="polite">
          Showing {visible.length} {filter === 'All' ? '' : filter.toLowerCase()} projects
        </p>

        <ul ref={grid} className="mt-16 columns-1 gap-6 sm:columns-2 md:gap-8 lg:columns-3">
          {projects.map((p) => {
            const shown = filter === 'All' || p.category === filter;
            const vIndex = visible.indexOf(p);
            return (
              <li key={p.id} data-item data-flip-id={p.id} className={cn('mb-10 break-inside-avoid', !shown && 'hidden')}>
                <figure>
                  <button
                    type="button"
                    data-cursor="View"
                    onClick={() => setOpen(vIndex)}
                    aria-label={`Open ${p.title}, ${p.place}`}
                    className="group relative block w-full overflow-hidden bg-sand"
                    style={{ aspectRatio: `${p.image.width} / ${p.image.height}`, borderRadius: p.shape === 'arch' ? '999px 999px 0 0' : '2px' }}
                  >
                    <SmartImage
                      src={p.image}
                      alt={p.alt}
                      sizes="(min-width: 1024px) 30vw, (min-width: 640px) 46vw, 92vw"
                      imgClassName="transition-transform duration-[1400ms] ease-out-expo group-hover:scale-[1.06] group-focus-visible:scale-[1.06]"
                    />
                    <span className="eyebrow absolute top-4 left-4 bg-ivory/92 px-3 py-1.5 text-[0.6rem] text-indigo" style={p.shape === 'arch' ? { top: 'auto', bottom: '1rem' } : undefined}>
                      {p.category}
                    </span>
                  </button>
                  <figcaption className="mt-4 flex items-baseline justify-between gap-4">
                    <span className="font-serif text-2xl leading-tight">{p.title}</span>
                    <span className="eyebrow shrink-0 text-ink-muted">
                      {p.place} · {p.year}
                    </span>
                  </figcaption>
                </figure>
              </li>
            );
          })}
        </ul>

        <StarRule className="mx-auto my-24 max-w-3xl" />
        <BeforeAfter />
      </div>

      {open !== null && visible[open] && <Lightbox items={visible} index={open} onClose={() => setOpen(null)} onIndex={setOpen} />}
    </section>
  );
}
