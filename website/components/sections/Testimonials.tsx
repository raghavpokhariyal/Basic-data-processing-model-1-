import { SplitReveal, Reveal } from '@/components/motion/Reveal';
import { Mandala } from '@/components/ornaments/Mandala';
import { testimonials } from '@/lib/content';

type Quote = (typeof testimonials)[number];

function Card({ t }: { t: Quote }) {
  return (
    <figure className="relative flex w-[19rem] shrink-0 flex-col rounded-t-[999px] border border-brass/35 bg-paper px-8 pt-16 pb-8 sm:w-[23rem]">
      <svg className="absolute top-8 left-1/2 h-6 w-8 -translate-x-1/2 text-brass" viewBox="0 0 32 24" fill="currentColor" aria-hidden="true">
        <path d="M0 24V14C0 6 4 1.5 12 0l1.5 3C9 4.5 7 7.5 7 11h6v13zm19 0V14c0-8 4-12.5 12-14l1.5 3C28 4.5 26 7.5 26 11h6v13z" opacity="0.7" />
      </svg>
      <blockquote className="mt-4 flex-1 text-center font-serif text-[1.35rem] leading-snug text-indigo italic">“{t.quote}”</blockquote>
      <figcaption className="mt-6 text-center">
        <span className="block text-sm font-semibold">{t.name}</span>
        <span className="eyebrow mt-1 block text-[0.62rem] text-terracotta-deep">{t.city}</span>
      </figcaption>
    </figure>
  );
}

function Row({ items, reverse, duration }: { items: Quote[]; reverse?: boolean; duration: number }) {
  return (
    <div className="marquee group/m relative overflow-x-auto [scrollbar-width:none] motion-safe:overflow-hidden [&::-webkit-scrollbar]:hidden">
      <div
        className="marquee-track flex w-max gap-6 py-2"
        data-direction={reverse ? 'reverse' : undefined}
        style={{ ['--marquee-duration' as string]: `${duration}s` }}
      >
        {items.map((t) => (
          <Card key={t.name} t={t} />
        ))}
        {/* Duplicate set for a seamless loop; hidden from assistive tech */}
        <div aria-hidden="true" className="flex gap-6 motion-reduce:hidden">
          {items.map((t) => (
            <Card key={`${t.name}-dup`} t={t} />
          ))}
        </div>
      </div>
    </div>
  );
}

export function Testimonials() {
  const half = Math.ceil(testimonials.length / 2);
  return (
    <section aria-labelledby="testimonials-title" className="relative overflow-hidden bg-ivory py-28 md:py-36">
      <Mandala className="pointer-events-none absolute top-1/2 left-1/2 h-[60rem] w-[60rem] -translate-x-1/2 -translate-y-1/2 text-brass/[0.09]" />
      <div className="container-x relative text-center">
        <Reveal>
          <p className="eyebrow text-terracotta-deep">
            Kind words <span lang="hi" className="text-[0.95rem] tracking-normal normal-case">· आपके शब्द</span>
          </p>
        </Reveal>
        <SplitReveal as="h2" id="testimonials-title" className="display-lg mx-auto mt-6 max-w-[16ch] font-serif">
          Homes, in the words of those who live in them.
        </SplitReveal>
      </div>
      <div className="relative mt-16 space-y-6 [mask-image:linear-gradient(90deg,transparent,#000_8%,#000_92%,transparent)]">
        <Row items={testimonials.slice(0, half)} duration={70} />
        <Row items={testimonials.slice(half)} duration={80} reverse />
      </div>
    </section>
  );
}
