'use client';

import { useRef, useState } from 'react';
import { gsap, ScrollTrigger, useGSAP, prefersReducedMotion } from '@/lib/gsap';
import { useApp } from '@/components/providers/AppProvider';
import { Monogram } from '@/components/ornaments/Monogram';
import { site } from '@/lib/site';

/**
 * First-load intro: the monogram is drawn as line art inside its arch,
 * then a terracotta curtain rises over the loader and lifts away to
 * reveal the site. Repeat visits in a session get only the curtain.
 */
export function Preloader() {
  const { markRevealed, lockScroll } = useApp();
  const root = useRef<HTMLDivElement>(null);
  const [done, setDone] = useState(false);

  useGSAP(
    () => {
      const el = root.current;
      if (!el) return;
      let seen = false;
      try {
        seen = sessionStorage.getItem('preloaded') === '1';
        sessionStorage.setItem('preloaded', '1');
      } catch {
        /* storage unavailable: always play the full intro */
      }
      lockScroll(true);
      const finish = () => {
        ScrollTrigger.refresh();
        setDone(true);
      };
      const release = () => {
        lockScroll(false);
        markRevealed();
      };

      if (prefersReducedMotion()) {
        gsap.set(el.querySelectorAll('[data-draw]'), { strokeDashoffset: 0 });
        gsap.set(el.querySelectorAll('[data-pl-in]'), { opacity: 1, y: 0 });
        gsap.to(el, { autoAlpha: 0, duration: 0.35, delay: 0.15, onStart: release, onComplete: finish });
        return;
      }

      const counter = el.querySelector<HTMLElement>('[data-count]');
      const fonts = Promise.race([document.fonts?.ready ?? Promise.resolve(), new Promise((r) => setTimeout(r, 2500))]);
      const tl = gsap.timeline({ defaults: { ease: 'power2.inOut' } });

      if (!seen) {
        const prog = { v: 0 };
        tl.to('[data-draw]', { strokeDashoffset: 0, duration: 1.25, stagger: 0.14 }, 0)
          .from('[data-dot]', { scale: 0, transformOrigin: '50% 50%', duration: 0.5, ease: 'back.out(3)' }, 0.95)
          .to('[data-pl-in]', { opacity: 1, y: 0, duration: 0.9, stagger: 0.08, ease: 'expo.out' }, 0.7)
          .to(prog, { v: 100, duration: 1.9, ease: 'power1.inOut', onUpdate: () => counter && (counter.textContent = String(Math.round(prog.v)).padStart(3, '0')) }, 0)
          .addPause('+=0.15', () => {
            fonts.then(() => tl.resume());
          });
      } else {
        tl.set('[data-draw]', { strokeDashoffset: 0 }).set('[data-pl-in]', { opacity: 1, y: 0 }).to({}, { duration: 0.2 });
      }

      tl.set('[data-panel]', { yPercent: 120 })
        .to('[data-panel]', { yPercent: 0, duration: 0.85, ease: 'expo.inOut' })
        .set(['[data-stage]', '[data-bg]'], { autoAlpha: 0 })
        .add(release)
        .to('[data-panel]', { yPercent: -100, duration: 1, ease: 'expo.inOut' })
        .add(finish);
    },
    { scope: root },
  );

  if (done) return null;

  return (
    <div ref={root} className="preloader fixed inset-0 z-[100]" role="status" aria-label={`Loading ${site.name}`}>
      <div data-bg className="absolute inset-0 bg-ivory" />
      <div data-stage className="absolute inset-0 grid place-items-center">
        <div className="flex flex-col items-center text-indigo">
          <Monogram className="h-36 w-auto text-terracotta sm:h-44 [&_[data-draw]]:[stroke-dasharray:1] [&_[data-draw]]:[stroke-dashoffset:1]" strokeWidth={1.6} />
          <p data-pl-in className="mt-8 translate-y-3 font-serif text-3xl tracking-[0.42em] uppercase opacity-0 sm:text-4xl">{site.name}</p>
          <p data-pl-in lang="hi" className="mt-2 translate-y-3 text-lg text-terracotta-deep opacity-0">
            {site.nameDevanagari}
          </p>
          <p data-pl-in className="eyebrow mt-6 translate-y-3 text-ink-muted opacity-0">{site.tagline}</p>
        </div>
        <span data-count className="eyebrow absolute bottom-8 left-1/2 -translate-x-1/2 tabular-nums text-ink-muted" aria-hidden="true">
          000
        </span>
      </div>
      <div data-panel className="absolute inset-0" style={{ transform: 'translateY(120%)' }} aria-hidden="true">
        <div className="absolute inset-x-0 -top-[16vh] h-[16.2vh] bg-terracotta" style={{ borderRadius: '50% 50% 0 0 / 100% 100% 0 0' }} />
        <div className="absolute inset-0 bg-terracotta" />
      </div>
    </div>
  );
}
