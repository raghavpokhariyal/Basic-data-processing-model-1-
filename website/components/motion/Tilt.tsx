'use client';

import { useRef, type ReactNode } from 'react';
import { gsap, useGSAP, MQ } from '@/lib/gsap';
import { cn } from '@/lib/cn';

/** Subtle 3D hover tilt for cards. */
export function Tilt({ children, max = 5, className }: { children: ReactNode; max?: number; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);

  useGSAP(
    () => {
      const el = ref.current;
      if (!el) return;
      const mm = gsap.matchMedia();
      mm.add(`${MQ.fine} and ${MQ.motion}`, () => {
        gsap.set(el, { transformPerspective: 1000, transformStyle: 'preserve-3d' });
        const rx = gsap.quickTo(el, 'rotationX', { duration: 0.8, ease: 'power3' });
        const ry = gsap.quickTo(el, 'rotationY', { duration: 0.8, ease: 'power3' });
        const move = (e: PointerEvent) => {
          const b = el.getBoundingClientRect();
          const px = (e.clientX - b.left) / b.width - 0.5;
          const py = (e.clientY - b.top) / b.height - 0.5;
          ry(px * max * 2);
          rx(-py * max * 2);
        };
        const leave = () => {
          rx(0);
          ry(0);
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
    <div ref={ref} className={cn('will-change-transform', className)}>
      {children}
    </div>
  );
}
