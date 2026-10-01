'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { gsap, ScrollTrigger, useGSAP, MQ, prefersReducedMotion } from '@/lib/gsap';
import { useApp } from '@/components/providers/AppProvider';
import { SmartImage } from '@/components/ui/SmartImage';
import { ArchButton } from '@/components/ui/ArchButton';
import { hero } from '@/lib/content';
import { cn } from '@/lib/cn';

const SLIDE_MS = 7000;

export function Hero() {
  const root = useRef<HTMLElement>(null);
  const { revealed, lenis } = useApp();
  const [index, setIndex] = useState(0);
  const indexRef = useRef(0);
  const intro = useRef<gsap.core.Timeline | null>(null);
  const progress = useRef<gsap.core.Tween | null>(null);
  const inView = useRef(true);
  const slides = hero.slides;

  // Intro timeline (built paused; played once the page is revealed)
  useGSAP(
    () => {
      const mm = gsap.matchMedia();
      mm.add(MQ.motion, () => {
        const tl = gsap.timeline({ paused: true, defaults: { ease: 'expo.out' } });
        tl.from('[data-hero-media]', { scale: 1.18, duration: 2.4, ease: 'expo.out' }, 0)
          .from('[data-word]', { yPercent: 120, rotate: 4, duration: 1.4, stagger: 0.075 }, 0.15)
          .from('[data-hero-fade]', { y: 26, opacity: 0, duration: 1.2, stagger: 0.1 }, 0.55)
          .from('[data-hero-line]', { scaleX: 0, transformOrigin: 'left center', duration: 1.4 }, 0.7);
        intro.current = tl;

        // Parallax as the hero scrolls away
        gsap.to('[data-hero-parallax]', {
          yPercent: 16,
          ease: 'none',
          scrollTrigger: { trigger: root.current, start: 'top top', end: 'bottom top', scrub: true },
        });
        gsap.to('[data-hero-content]', {
          yPercent: -18,
          opacity: 0.15,
          ease: 'none',
          scrollTrigger: { trigger: root.current, start: 'top top', end: 'bottom top', scrub: true },
        });
        return () => {
          intro.current = null;
        };
      });
      ScrollTrigger.create({
        trigger: root.current,
        start: 'top bottom',
        end: 'bottom top',
        onToggle: (self) => {
          inView.current = self.isActive;
          if (self.isActive) progress.current?.resume();
          else progress.current?.pause();
        },
      });
    },
    { scope: root },
  );

  useEffect(() => {
    if (revealed) intro.current?.play();
  }, [revealed]);

  const show = useCallback(
    (next: number) => {
      const el = root.current;
      if (!el || next === indexRef.current) return;
      const slidesEl = el.querySelectorAll<HTMLElement>('[data-slide]');
      const prev = slidesEl[indexRef.current];
      const incoming = slidesEl[next];
      indexRef.current = next;
      setIndex(next);
      const reduce = prefersReducedMotion();
      gsap.killTweensOf([prev, incoming]);
      gsap.set(slidesEl, { zIndex: 0 });
      gsap.set(prev, { zIndex: 1 });
      gsap.set(incoming, { zIndex: 2, autoAlpha: 1 });
      if (reduce) {
        gsap.set(incoming, { clipPath: 'inset(0% 0% 0% 0%)' });
        gsap.set(prev, { autoAlpha: 0 });
        return;
      }
      const img = incoming.querySelector('[data-kenburns]');
      gsap.fromTo(incoming, { clipPath: 'inset(0% 0% 100% 0%)' }, { clipPath: 'inset(0% 0% 0% 0%)', duration: 1.6, ease: 'expo.inOut', onComplete: () => gsap.set(prev, { autoAlpha: 0 }) });
      gsap.fromTo(img, { scale: 1.16 }, { scale: 1.0, duration: SLIDE_MS / 1000 + 1.6, ease: 'power1.out' });
    },
    [],
  );

  // Autoplay with a progress bar; pauses off-screen and when the tab is hidden
  useEffect(() => {
    if (!revealed || prefersReducedMotion()) return;
    const bar = root.current?.querySelector('[data-progress]');
    const run = () => {
      progress.current?.kill();
      progress.current = gsap.fromTo(
        bar ?? {},
        { scaleX: 0 },
        {
          scaleX: 1,
          duration: SLIDE_MS / 1000,
          ease: 'none',
          onComplete: () => {
            show((indexRef.current + 1) % slides.length);
            run();
          },
        },
      );
      if (!inView.current || document.hidden) progress.current.pause();
    };
    run();
    const vis = () => (document.hidden ? progress.current?.pause() : inView.current && progress.current?.resume());
    document.addEventListener('visibilitychange', vis);
    return () => {
      document.removeEventListener('visibilitychange', vis);
      progress.current?.kill();
    };
  }, [revealed, show, slides.length]);

  const pick = (i: number) => {
    show(i);
    progress.current?.restart();
  };

  const current = slides[index];

  return (
    <section ref={root} id="top" data-hero data-theme="dark" aria-label="Introduction" className="relative h-[100svh] min-h-[40rem] overflow-hidden bg-indigo-deep text-ivory">
      <div data-hero-parallax className="absolute inset-0 -bottom-[12%]">
        <div data-hero-media className="absolute inset-0">
          {slides.map((s, i) => (
            <div
              key={s.title}
              data-slide
              className="absolute inset-0"
              style={{ zIndex: i === 0 ? 2 : 0, visibility: i === 0 ? 'visible' : 'hidden' }}
              aria-hidden={i !== index}
            >
              <div data-kenburns className="absolute inset-0">
                <SmartImage src={s.image} alt={s.alt} sizes="100vw" preload={i === 0} loading={i === 0 ? 'eager' : 'lazy'} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Legibility washes */}
      <div aria-hidden="true" className="absolute inset-0 bg-gradient-to-t from-indigo-deep/90 via-indigo-deep/35 to-indigo-deep/10" />
      <div aria-hidden="true" className="absolute inset-0 bg-gradient-to-r from-indigo-deep/60 via-transparent to-transparent" />

      <div data-hero-content className="container-x relative z-10 flex h-full flex-col justify-end pb-28 sm:pb-32">
        <p data-hero-fade className="eyebrow text-sand">
          {hero.eyebrow}
        </p>
        <p data-hero-fade lang="hi" aria-hidden="true" className="mt-2 text-base text-sand/80">
          निर्माण · सज्जा · शिल्प
        </p>
        <h1 className="display-xl mt-6 font-serif font-medium" aria-label={hero.lines.map((l) => l.join(' ')).join(' ')}>
          {hero.lines.map((line, li) => (
            <span key={li} className="block" aria-hidden="true">
              {line.map((w, wi) => (
                <span key={wi} className="-mb-[0.14em] inline-block overflow-hidden pb-[0.14em] align-bottom">
                  <span data-word className={cn('inline-block', li === 1 && wi === line.length - 1 && 'text-terracotta-light italic')}>
                    {w}
                  </span>
                  {wi < line.length - 1 && <span className="inline-block">&nbsp;</span>}
                </span>
              ))}
            </span>
          ))}
        </h1>
        <div className="mt-8 grid gap-8 md:grid-cols-[minmax(0,34rem)_1fr] md:items-end">
          <p data-hero-fade className="max-w-xl text-base leading-relaxed text-ivory/90 sm:text-lg">
            {hero.sub}
          </p>
          <div data-hero-fade className="flex flex-wrap gap-3 md:justify-end">
            <ArchButton href="/#contact" variant="light">
              Begin your project
            </ArchButton>
            <ArchButton href="/#work" variant="outline-light">
              View our work
            </ArchButton>
          </div>
        </div>
      </div>

      {/* Slide caption, controls and progress */}
      <div className="container-x absolute inset-x-0 bottom-0 z-10 pb-6 sm:pb-8">
        <div className="relative" aria-hidden="true">
          <div data-hero-line className="h-px w-full bg-ivory/25" />
          <div data-progress className="absolute inset-0 h-px origin-left scale-x-0 bg-terracotta-light" />
        </div>
        <div className="mt-4 flex items-center justify-between gap-6 text-xs">
          <p data-hero-fade className="min-w-0 truncate text-ivory/85" aria-live="polite">
            <span className="tabular-nums text-sand">{String(index + 1).padStart(2, '0')} / {String(slides.length).padStart(2, '0')}</span>
            <span className="mx-3 text-ivory/40">—</span>
            <span className="font-serif text-base italic">{current.title}</span>
            <span className="text-ivory/60">, {current.place}</span>
          </p>
          <div data-hero-fade className="flex shrink-0 items-center gap-2">
            {slides.map((s, i) => (
              <button
                key={s.title}
                type="button"
                onClick={() => pick(i)}
                aria-label={`Show slide ${i + 1}: ${s.title}`}
                aria-current={i === index}
                className="group grid h-8 w-8 place-items-center"
              >
                <span className={cn('block h-2 w-2 rounded-full border border-ivory/70 transition-all duration-500', i === index ? 'scale-125 bg-ivory' : 'group-hover:bg-ivory/50')} />
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Scroll cue: a small arch with a falling bead */}
      <button
        type="button"
        data-hero-fade
        onClick={() => {
          const target = document.getElementById('about');
          if (!target) return;
          if (lenis) lenis.scrollTo(target, { duration: 1.6 });
          else target.scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
        }}
        className="absolute top-[38%] right-[clamp(1rem,4vw,3.5rem)] z-10 hidden flex-col items-center gap-3 text-ivory/80 md:flex"
        aria-label="Scroll to the next section"
      >
        <span className="relative block h-11 w-7 rounded-t-full border border-ivory/60">
          <span className="absolute top-2 left-1/2 -ml-[2px] block h-1.5 w-1 rounded-full bg-ivory [animation:scroll-cue_2.2s_cubic-bezier(0.65,0.05,0.36,1)_infinite]" />
        </span>
        <span className="eyebrow text-[0.6rem] [writing-mode:vertical-rl]">Scroll</span>
      </button>
    </section>
  );
}
