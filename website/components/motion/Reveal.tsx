'use client';

import { useRef, type ElementType, type ReactNode } from 'react';
import { gsap, SplitText, useGSAP, MQ } from '@/lib/gsap';
import { cn } from '@/lib/cn';

/**
 * Splits a heading into masked lines and words, revealing them word by
 * word as it scrolls into view. Screen readers get the full text.
 */
export function SplitReveal({
  as: Tag = 'h2',
  children,
  className,
  id,
  stagger = 0.05,
  delay = 0,
  start = 'top 86%',
}: {
  as?: ElementType;
  children: ReactNode;
  className?: string;
  id?: string;
  stagger?: number;
  delay?: number;
  start?: string;
}) {
  const ref = useRef<HTMLElement>(null);

  useGSAP(
    () => {
      const el = ref.current;
      if (!el) return;
      const mm = gsap.matchMedia();
      mm.add(MQ.motion, () => {
        const split = SplitText.create(el, {
          type: 'lines,words',
          mask: 'lines',
          autoSplit: true,
          onSplit(self) {
            return gsap.from(self.words, {
              yPercent: 115,
              rotate: 2,
              duration: 1.15,
              ease: 'expo.out',
              stagger,
              delay,
              scrollTrigger: { trigger: el, start, once: true },
            });
          },
        });
        return () => split.revert();
      });
    },
    { scope: ref },
  );

  return (
    <Tag ref={ref} id={id} className={className}>
      {children}
    </Tag>
  );
}

/** Fades and lifts `[data-reveal]` descendants (or direct children) in sequence. */
export function Reveal({
  children,
  className,
  as: Tag = 'div',
  y = 36,
  stagger = 0.1,
  start = 'top 85%',
}: {
  children: ReactNode;
  className?: string;
  as?: ElementType;
  y?: number;
  stagger?: number;
  start?: string;
}) {
  const ref = useRef<HTMLElement>(null);

  useGSAP(
    () => {
      const el = ref.current;
      if (!el) return;
      const mm = gsap.matchMedia();
      mm.add(MQ.motion, () => {
        const marked = el.querySelectorAll('[data-reveal]');
        const targets = marked.length ? marked : el.children;
        gsap.from(targets, {
          y,
          opacity: 0,
          duration: 1.1,
          ease: 'expo.out',
          stagger,
          scrollTrigger: { trigger: el, start, once: true },
        });
      });
    },
    { scope: ref },
  );

  return (
    <Tag ref={ref} className={className}>
      {children}
    </Tag>
  );
}

/** Scroll-linked vertical drift for layered depth. */
export function Parallax({ children, className, speed = 12 }: { children: ReactNode; className?: string; speed?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  useGSAP(
    () => {
      const el = ref.current;
      if (!el) return;
      const mm = gsap.matchMedia();
      mm.add(MQ.motion, () => {
        gsap.fromTo(
          el,
          { yPercent: -speed / 2 },
          { yPercent: speed / 2, ease: 'none', scrollTrigger: { trigger: el, start: 'top bottom', end: 'bottom top', scrub: true } },
        );
      });
    },
    { scope: ref },
  );
  return (
    <div ref={ref} className={cn('will-change-transform', className)}>
      {children}
    </div>
  );
}

/** Counts up to `value` when scrolled into view (Indian digit grouping). */
export function Counter({ value, suffix = '', className }: { value: number; suffix?: string; className?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  useGSAP(
    () => {
      const el = ref.current;
      if (!el) return;
      const mm = gsap.matchMedia();
      mm.add(MQ.motion, () => {
        const n = { v: 0 };
        const fmt = (v: number) => Math.round(v).toLocaleString('en-IN') + suffix;
        el.textContent = fmt(0);
        gsap.to(n, {
          v: value,
          duration: 2.2,
          ease: 'power2.out',
          scrollTrigger: { trigger: el, start: 'top 90%', once: true },
          onUpdate: () => {
            el.textContent = fmt(n.v);
          },
        });
        return () => {
          el.textContent = fmt(value);
        };
      });
    },
    { scope: ref },
  );
  return (
    <span ref={ref} className={cn('tabular-nums', className)}>
      {value.toLocaleString('en-IN') + suffix}
    </span>
  );
}
