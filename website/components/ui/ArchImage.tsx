'use client';

import type { StaticImageData } from 'next/image';
import { useId, useRef } from 'react';
import { gsap, useGSAP, MQ } from '@/lib/gsap';
import { SmartImage } from './SmartImage';
import { cn } from '@/lib/cn';

type Shape = 'round' | 'mughal' | 'cusped' | 'rect';
type RevealMode = 'wipe' | 'jaali' | 'none';

const shapeClass: Record<Shape, string> = {
  round: 'arch',
  mughal: 'arch-mughal',
  cusped: 'arch-cusped',
  rect: '',
};

/**
 * An image masked into an arch. On scroll it is revealed either by a
 * vertical wipe or through a jaali lattice whose bars dissolve, and it
 * drifts slightly inside its frame for parallax.
 */
export function ArchImage({
  src,
  alt,
  sizes,
  shape = 'round',
  reveal = 'wipe',
  parallax = true,
  tone = 'ivory',
  className,
  preload,
}: {
  src: StaticImageData;
  alt: string;
  sizes: string;
  shape?: Shape;
  reveal?: RevealMode;
  parallax?: boolean;
  tone?: 'ivory' | 'sand' | 'indigo';
  className?: string;
  preload?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const patternId = `jaali-${useId().replace(/[^a-zA-Z0-9]/g, '')}`;

  useGSAP(
    () => {
      const el = ref.current;
      if (!el) return;
      const mm = gsap.matchMedia();
      mm.add(MQ.motion, () => {
        const inner = el.querySelector('[data-wipe]');
        const media = el.querySelector('[data-media]');
        const tl = gsap.timeline({ scrollTrigger: { trigger: el, start: 'top 82%', once: true } });
        if (reveal === 'wipe') {
          tl.fromTo(inner, { clipPath: 'inset(100% 0% 0% 0%)' }, { clipPath: 'inset(0% 0% 0% 0%)', duration: 1.5, ease: 'expo.inOut' }).from(
            media,
            { scale: 1.3, duration: 1.8, ease: 'expo.out' },
            0.1,
          );
        } else if (reveal === 'jaali') {
          const bars = el.querySelectorAll('[data-bar]');
          gsap.set(el.querySelector('[data-jaali]'), { opacity: 1 });
          tl.fromTo(bars, { attr: { 'stroke-width': 30 } }, { attr: { 'stroke-width': 0 }, duration: 1.8, ease: 'power3.inOut' })
            .to(el.querySelector('[data-jaali]'), { opacity: 0, duration: 0.4 }, '-=0.35')
            .from(media, { scale: 1.18, duration: 2, ease: 'expo.out' }, 0);
        }
        if (parallax && media) {
          gsap.fromTo(media, { yPercent: -5 }, { yPercent: 5, ease: 'none', scrollTrigger: { trigger: el, start: 'top bottom', end: 'bottom top', scrub: true } });
        }
      });
    },
    { scope: ref },
  );

  const fill = tone === 'sand' ? 'var(--color-sand-light)' : tone === 'indigo' ? 'var(--color-indigo-deep)' : 'var(--color-ivory)';

  return (
    <div ref={ref} className={cn('relative isolate overflow-hidden', shapeClass[shape], className)}>
      <div data-wipe className="absolute inset-0">
        <div data-media className="absolute inset-x-0 -inset-y-[6%]">
          <SmartImage src={src} alt={alt} sizes={sizes} preload={preload} />
        </div>
      </div>
      {reveal === 'jaali' && (
        <svg data-jaali className="pointer-events-none absolute inset-0 h-full w-full opacity-0" aria-hidden="true" focusable="false">
          <defs>
            <pattern id={patternId} width="56" height="56" patternUnits="userSpaceOnUse">
              <g fill="none" stroke={fill} strokeWidth="30" data-bar>
                <path data-bar d="M28 6l6.4 15.6L50 28l-15.6 6.4L28 50l-6.4-15.6L6 28l15.6-6.4z" />
                <path data-bar d="M0 0L56 56M56 0L0 56" />
                <rect data-bar x="0" y="0" width="56" height="56" />
              </g>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill={`url(#${patternId})`} />
        </svg>
      )}
    </div>
  );
}
