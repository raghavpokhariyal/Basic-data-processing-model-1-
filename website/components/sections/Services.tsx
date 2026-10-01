import { SplitReveal, Reveal } from '@/components/motion/Reveal';
import { Tilt } from '@/components/motion/Tilt';
import { TransitionLink } from '@/components/motion/TransitionLink';
import { SmartImage } from '@/components/ui/SmartImage';
import { Arrow } from '@/components/ui/ArchButton';
import { JaaliDivider } from '@/components/ornaments/Jaali';
import { services } from '@/lib/content';
import { cn } from '@/lib/cn';

export function Services() {
  return (
    <section id="services" aria-labelledby="services-title" className="bg-blockprint relative bg-sand-light">
      <JaaliDivider tone="terracotta" className="opacity-60" />
      <div className="container-x py-28 md:py-36">
        <div className="grid gap-10 lg:grid-cols-12 lg:items-end">
          <div className="lg:col-span-7">
            <Reveal>
              <p className="eyebrow flex items-center gap-3 text-terracotta-deep">
                <span className="h-px w-10 bg-current" aria-hidden="true" />
                What we do
                <span lang="hi" className="text-[0.95rem] tracking-normal normal-case">
                  · हमारी सेवाएँ
                </span>
              </p>
            </Reveal>
            <SplitReveal as="h2" id="services-title" className="display-lg mt-7 max-w-[14ch] font-serif">
              From foundation to final furnishing.
            </SplitReveal>
          </div>
          <Reveal className="lg:col-span-4 lg:col-start-9">
            <p className="text-lg leading-relaxed text-ink-muted">
              Three connected disciplines under one roof. Engage us for one, or let a single team carry your home from the first drawing to the last cushion.
            </p>
          </Reveal>
        </div>

        <Reveal as="ul" className="mt-20 grid gap-12 md:grid-cols-3 md:gap-6 lg:gap-10" stagger={0.15}>
          {services.map((s, i) => (
            <li key={s.slug} className={cn('relative', i === 1 && 'md:mt-24', i === 2 && 'md:mt-10')}>
              <Tilt max={4}>
                <TransitionLink href={`/services/${s.slug}`} data-cursor="Explore" className="group block">
                  <div className="arch-mughal relative aspect-[3/4] overflow-hidden bg-sand">
                    <SmartImage
                      src={s.image}
                      alt={s.imageAlt}
                      sizes="(min-width: 768px) 31vw, 92vw"
                      imgClassName="transition-transform duration-[1600ms] ease-out-expo group-hover:scale-[1.09] group-focus-visible:scale-[1.09]"
                    />
                    <div aria-hidden="true" className="absolute inset-0 bg-gradient-to-t from-indigo-deep/90 via-indigo-deep/30 to-transparent transition-opacity duration-700 group-hover:from-indigo-deep/92" />
                    <div className="absolute inset-x-0 bottom-0 p-6 text-ivory sm:p-8">
                      <p lang="hi" aria-hidden="true" className="mb-3 font-deva text-[clamp(2.4rem,4vw,3.4rem)] leading-none text-terracotta-light">
                        {s.hindi}
                      </p>
                      <p className="eyebrow text-sand">
                        {s.number} <span className="mx-2 opacity-60">·</span> {s.transliteration}
                      </p>
                      <h3 className="mt-3 font-serif text-[2.4rem] leading-none">{s.title}</h3>
                      <div className="grid grid-rows-[0fr] transition-[grid-template-rows] duration-700 ease-out-expo group-hover:grid-rows-[1fr] group-focus-visible:grid-rows-[1fr] [@media(hover:none)]:grid-rows-[1fr]">
                        <div className="overflow-hidden">
                          <p className="pt-4 text-sm leading-relaxed text-ivory/90">{s.summary}</p>
                        </div>
                      </div>
                      <span className="eyebrow mt-5 inline-flex items-center gap-2 text-sand">
                        Explore {s.slug}
                        <Arrow className="transition-transform duration-500 group-hover:translate-x-1.5" />
                      </span>
                    </div>
                  </div>
                </TransitionLink>
              </Tilt>
              <p className="mt-6 max-w-xs font-serif text-xl text-indigo italic">{s.tagline}</p>
            </li>
          ))}
        </Reveal>
      </div>
    </section>
  );
}
