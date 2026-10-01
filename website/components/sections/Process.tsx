'use client';

import { useRef, useState } from 'react';
import { gsap, useGSAP, MQ } from '@/lib/gsap';
import { ArchImage } from '@/components/ui/ArchImage';
import { Mandala } from '@/components/ornaments/Mandala';
import { journey } from '@/lib/content';
import { cn } from '@/lib/cn';

/**
 * Consultation → Design → Build → Furnish → Handover.
 * On tablets and desktops the section pins and the steps travel
 * horizontally with the scroll; elsewhere it is a vertical timeline.
 */
export function Process() {
  const root = useRef<HTMLElement>(null);
  const track = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(-1);

  useGSAP(
    () => {
      const mm = gsap.matchMedia();
      mm.add(`${MQ.desktop} and ${MQ.motion}`, () => {
        const t = track.current;
        if (!t) return;
        const distance = () => Math.max(0, t.scrollWidth - window.innerWidth);
        const tween = gsap.to(t, {
          x: () => -distance(),
          ease: 'none',
          scrollTrigger: {
            trigger: root.current,
            start: 'top top',
            end: () => `+=${distance()}`,
            pin: true,
            scrub: 0.8,
            invalidateOnRefresh: true,
            anticipatePin: 1,
          },
        });
        gsap.fromTo(
          '[data-progress-fill]',
          { scaleX: 0 },
          { scaleX: 1, ease: 'none', scrollTrigger: { trigger: root.current, start: 'top top', end: () => `+=${distance()}`, scrub: true, invalidateOnRefresh: true } },
        );
        gsap.utils.toArray<HTMLElement>('[data-step]').forEach((step, i) => {
          gsap.from(step.querySelectorAll('[data-step-in]'), {
            y: 70,
            opacity: 0,
            stagger: 0.08,
            duration: 1.1,
            ease: 'expo.out',
            scrollTrigger: { trigger: step, containerAnimation: tween, start: 'left 78%', toggleActions: 'play none none reverse' },
          });
          gsap.fromTo(
            step.querySelector('[data-numeral]'),
            { xPercent: 30 },
            { xPercent: -30, ease: 'none', scrollTrigger: { trigger: step, containerAnimation: tween, start: 'left right', end: 'right left', scrub: true } },
          );
          gsap.timeline({
            scrollTrigger: {
              trigger: step,
              containerAnimation: tween,
              start: 'left 55%',
              end: 'right 55%',
              onToggle: (self) => self.isActive && setActive(i),
              onLeaveBack: () => i === 0 && setActive(-1),
            },
          });
        });
      });
      mm.add(`not all and ${MQ.desktop}, ${MQ.reduce}`, () => {
        gsap.utils.toArray<HTMLElement>('[data-step]').forEach((step) => {
          gsap.from(step.querySelectorAll('[data-step-in]'), {
            y: 40,
            opacity: 0,
            stagger: 0.08,
            duration: 1,
            ease: 'expo.out',
            scrollTrigger: { trigger: step, start: 'top 82%', once: true },
          });
        });
      });
    },
    { scope: root },
  );

  return (
    <section ref={root} id="process" aria-labelledby="process-title" data-theme="dark" className="relative overflow-hidden bg-indigo-deep text-ivory pin:h-[100svh]">
      <Mandala className="pointer-events-none absolute -bottom-72 -left-64 h-[52rem] w-[52rem] text-brass/[0.12]" />

      <div ref={track} className="relative flex flex-col pin:h-full pin:w-max pin:flex-row pin:items-center">
        {/* Intro panel */}
        <div className="container-x flex shrink-0 flex-col justify-center pt-28 pb-16 pin:mx-0 pin:w-[44vw] pin:max-w-none pin:py-0 pin:pr-16 pin:pl-[max(4vw,3.5rem)]">
          <p className="eyebrow flex items-center gap-3 text-brass-light">
            <span className="h-px w-10 bg-current" aria-hidden="true" />
            {journey.eyebrow}
            <span lang="hi" className="text-[0.95rem] tracking-normal normal-case">
              · {journey.eyebrowHi}
            </span>
          </p>
          <h2 id="process-title" className="display-lg mt-7 max-w-[12ch] font-serif">
            {journey.heading}
          </h2>
          <p className="mt-8 max-w-md text-lg leading-relaxed text-sand">{journey.intro}</p>
          <p className="eyebrow mt-12 hidden items-center gap-3 text-sand/80 pin:flex" aria-hidden="true">
            Scroll to walk through
            <svg width="40" height="10" viewBox="0 0 40 10" fill="none" stroke="currentColor">
              <path d="M0 5h38M33 1l5 4-5 4" />
            </svg>
          </p>
        </div>

        {/* Steps */}
        <ol className="relative flex flex-col pin:h-full pin:flex-row pin:items-center">
          {journey.steps.map((step, i) => (
            <li
              key={step.title}
              data-step
              className="container-x relative grid shrink-0 gap-8 border-l border-brass/25 py-12 pl-8 pin:mx-0 pin:h-full pin:w-[min(76vw,64rem)] pin:max-w-none pin:grid-cols-[minmax(0,0.9fr)_minmax(0,1fr)] pin:items-center pin:gap-14 pin:border-l-0 pin:px-14 pin:py-0"
            >
              <span aria-hidden="true" className="absolute top-14 -left-[5px] h-[9px] w-[9px] rotate-45 bg-brass pin:hidden" />
              <span
                data-numeral
                aria-hidden="true"
                className="pointer-events-none absolute top-4 right-4 hidden font-serif text-[16rem] leading-none text-transparent [-webkit-text-stroke:1px_rgba(209,180,131,0.32)] pin:block"
              >
                0{i + 1}
              </span>
              <div data-step-in className="relative w-full max-w-sm pin:w-auto pin:max-w-none pin:justify-self-end">
                <ArchImage
                  src={step.image}
                  alt={step.alt}
                  sizes="(min-width: 768px) 30vw, 85vw"
                  shape={i % 2 ? 'mughal' : 'round'}
                  reveal="none"
                  parallax={false}
                  tone="indigo"
                  className="aspect-[4/5] w-full pin:h-[62svh] pin:w-auto"
                />
              </div>
              <div className="relative">
                <p data-step-in className="eyebrow text-brass-light">
                  Step 0{i + 1} <span className="mx-2 opacity-60">·</span> {step.time}
                </p>
                <p data-step-in lang="hi" aria-hidden="true" className="mt-5 text-3xl text-terracotta-light">
                  {step.hindi}
                </p>
                <h3 data-step-in className="mt-2 font-serif text-[clamp(2.6rem,4.4vw,4.25rem)] leading-none">
                  {step.title}
                </h3>
                <p data-step-in className="mt-6 max-w-md text-base leading-relaxed text-sand sm:text-lg">
                  {step.text}
                </p>
              </div>
            </li>
          ))}
          <li aria-hidden="true" className="hidden shrink-0 pin:block pin:w-[12vw]" />
        </ol>
      </div>

      {/* Progress rail (pinned layout only) */}
      <div className="container-x pointer-events-none absolute inset-x-0 bottom-8 hidden pin:block" aria-hidden="true">
        <div className="relative h-px bg-ivory/15">
          <div data-progress-fill className="absolute inset-0 origin-left scale-x-0 bg-brass" />
        </div>
        <ol className="mt-4 grid grid-cols-5 text-[0.7rem] font-semibold tracking-[0.16em] uppercase">
          {journey.steps.map((s, i) => (
            <li key={s.title} className={cn('flex items-center gap-2 transition-colors duration-500', i <= active ? 'text-ivory' : 'text-ivory/40')}>
              <span className={cn('h-1.5 w-1.5 rotate-45 transition-colors duration-500', i <= active ? 'bg-brass' : 'bg-ivory/30')} />
              {s.title}
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
