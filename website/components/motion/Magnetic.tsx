'use client';

import { useRef, type ReactNode } from 'react';
import { gsap, useGSAP, MQ } from '@/lib/gsap';
import { cn } from '@/lib/cn';

/** Pulls its child gently toward the pointer; springs back on leave. */
export function Magnetic({ children, strength = 0.32, className }: { children: ReactNode; strength?: number; className?: string }) {
  const ref = useRef<HTMLSpanElement>(null);

  useGSAP(
    () => {
      const el = ref.current;
      if (!el) return;
      const mm = gsap.matchMedia();
      mm.add(`${MQ.fine} and ${MQ.motion}`, () => {
        const inner = el.firstElementChild as HTMLElement | null;
        const xTo = gsap.quickTo(el, 'x', { duration: 0.6, ease: 'power3' });
        const yTo = gsap.quickTo(el, 'y', { duration: 0.6, ease: 'power3' });
        const move = (e: PointerEvent) => {
          const b = el.getBoundingClientRect();
          const x = e.clientX - (b.left + b.width / 2);
          const y = e.clientY - (b.top + b.height / 2);
          xTo(x * strength);
          yTo(y * strength);
          if (inner) gsap.to(inner, { x: x * strength * 0.35, y: y * strength * 0.35, duration: 0.6, ease: 'power3' });
        };
        const leave = () => {
          gsap.to(el, { x: 0, y: 0, duration: 0.9, ease: 'elastic.out(1, 0.4)' });
          if (inner) gsap.to(inner, { x: 0, y: 0, duration: 0.9, ease: 'elastic.out(1, 0.4)' });
        };
        el.addEventListener('pointermove', move);
        el.addEventListener('pointerleave', leave);
        return () => {
          el.removeEventListener('pointermove', move);
          el.removeEventListener('pointerleave', leave);
        };
      });
    },
    { scope: ref },
  );

  return (
    <span ref={ref} className={cn('inline-block will-change-transform', className)}>
      {children}
    </span>
  );
}
