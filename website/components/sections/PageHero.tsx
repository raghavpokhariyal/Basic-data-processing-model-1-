'use client';

import type { StaticImageData } from 'next/image';
import { useEffect, useRef } from 'react';
import { gsap, useGSAP, MQ } from '@/lib/gsap';
import { useApp } from '@/components/providers/AppProvider';
import { SmartImage } from '@/components/ui/SmartImage';
import { TransitionLink } from '@/components/motion/TransitionLink';

/** Hero for inner pages; its intro plays as the curtain lifts. */
export function PageHero({
  eyebrow,
  title,
  hindi,
  lead,
  image,
  alt,
  crumbs,
}: {
  eyebrow: string;
  title: string;
  hindi?: string;
  lead: string;
  image: StaticImageData;
  alt: string;
  crumbs: { label: string; href?: string }[];
}) {
  const root = useRef<HTMLElement>(null);
  const tl = useRef<gsap.core.Timeline | null>(null);
  const { revealed } = useApp();

  useGSAP(
    () => {
      const mm = gsap.matchMedia();
      mm.add(MQ.motion, () => {
        tl.current = gsap
          .timeline({ paused: true, defaults: { ease: 'expo.out' } })
          .from('[data-ph-media]', { scale: 1.2, duration: 2.2 }, 0)
          .from('[data-word]', { yPercent: 120, rotate: 3, duration: 1.3, stagger: 0.08 }, 0.15)
          .from('[data-ph-fade]', { y: 24, opacity: 0, duration: 1.1, stagger: 0.08 }, 0.45);
        gsap.to('[data-ph-media]', { yPercent: 14, ease: 'none', scrollTrigger: { trigger: root.current, start: 'top top', end: 'bottom top', scrub: true } });
        return () => {
          tl.current = null;
        };
      });
    },
    { scope: root },
  );

  useEffect(() => {
    if (revealed) tl.current?.play();
  }, [revealed]);

  return (
    <section ref={root} id="top" data-hero data-theme="dark" className="relative flex min-h-[38rem] items-end overflow-hidden bg-indigo-deep text-ivory h-[86svh]">
      <div data-ph-media className="absolute inset-0 -bottom-[12%]">
        <SmartImage src={image} alt={alt} sizes="100vw" preload />
      </div>
      <div aria-hidden="true" className="absolute inset-0 bg-gradient-to-t from-indigo-deep/92 via-indigo-deep/40 to-indigo-deep/25" />
      <div className="container-x relative z-10 pb-16 sm:pb-20">
        <nav data-ph-fade aria-label="Breadcrumb">
          <ol className="eyebrow flex flex-wrap items-center gap-2 text-sand">
            {crumbs.map((c, i) => (
              <li key={c.label} className="flex items-center gap-2">
                {c.href ? (
                  <TransitionLink href={c.href} className="link-underline">
                    {c.label}
                  </TransitionLink>
                ) : (
                  <span aria-current="page" className="text-ivory">
                    {c.label}
                  </span>
                )}
                {i < crumbs.length - 1 && <span aria-hidden="true">/</span>}
              </li>
            ))}
          </ol>
        </nav>
        <p data-ph-fade className="eyebrow mt-10 text-sand">
          {eyebrow}
        </p>
        <h1 className="display-xl mt-4 font-serif" aria-label={title}>
          {title.split(' ').map((w, i) => (
            <span key={i} aria-hidden="true" className="-mb-[0.14em] inline-block overflow-hidden pb-[0.14em] align-bottom">
              <span data-word className="inline-block">
                {w}
              </span>
              <span className="inline-block">&nbsp;</span>
            </span>
          ))}
          {hindi && (
            <span data-ph-fade lang="hi" aria-hidden="true" className="mt-2 block font-deva text-[0.42em] text-terracotta-light">
              {hindi}
            </span>
          )}
        </h1>
        <p data-ph-fade className="mt-8 max-w-2xl text-lg leading-relaxed text-ivory/90">
          {lead}
        </p>
      </div>
    </section>
  );
}
