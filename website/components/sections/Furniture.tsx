import { SplitReveal, Reveal } from '@/components/motion/Reveal';
import { ArchButton } from '@/components/ui/ArchButton';
import { SmartImage } from '@/components/ui/SmartImage';
import { StarRule } from '@/components/ornaments/Jaali';
import { ChairViewer } from './ChairViewer';
import { ProductCarousel } from './ProductCarousel';
import { featuredPiece, materials } from '@/lib/content';

export function Furniture() {
  return (
    <section id="furniture" aria-labelledby="furniture-title" className="relative overflow-hidden bg-ivory-deep py-28 md:py-40">
      <div className="container-x">
        <div className="grid gap-10 lg:grid-cols-12 lg:items-end">
          <div className="lg:col-span-7">
            <Reveal>
              <p className="eyebrow flex items-center gap-3 text-terracotta-deep">
                <span className="h-px w-10 bg-current" aria-hidden="true" />
                The workshop
                <span lang="hi" className="text-[0.95rem] tracking-normal normal-case">
                  · शिल्पशाला
                </span>
              </p>
            </Reveal>
            <SplitReveal as="h2" id="furniture-title" className="display-lg mt-7 max-w-[13ch] font-serif">
              Furniture with a maker’s mark.
            </SplitReveal>
          </div>
          <Reveal className="lg:col-span-4 lg:col-start-9">
            <p className="text-lg leading-relaxed text-ink-muted">
              Every piece is drawn in our studio and made by hand in our workshop, in solid timber, cane and brass. Turn the chair below to see how it is put together.
            </p>
          </Reveal>
        </div>

        <div className="mt-20 grid gap-14 lg:grid-cols-12 lg:items-center">
          <div className="lg:col-span-7">
            <ChairViewer />
          </div>
          <Reveal className="lg:col-span-4 lg:col-start-9">
            <p data-reveal className="eyebrow text-terracotta-deep">
              Featured piece
            </p>
            <h3 data-reveal className="mt-4 font-serif text-[clamp(2.4rem,4vw,3.6rem)] leading-none">
              {featuredPiece.name}
            </h3>
            <p data-reveal lang="hi" aria-hidden="true" className="mt-2 text-2xl text-terracotta">
              {featuredPiece.hindi}
            </p>
            <p data-reveal className="mt-6 text-lg leading-relaxed text-ink-muted">
              {featuredPiece.text}
            </p>
            <dl data-reveal className="mt-8 divide-y divide-brass/30 border-y border-brass/30">
              {featuredPiece.specs.map((s) => (
                <div key={s.label} className="flex justify-between gap-6 py-3 text-sm">
                  <dt className="text-ink-muted">{s.label}</dt>
                  <dd className="text-right font-semibold">{s.value}</dd>
                </div>
              ))}
            </dl>
            <div data-reveal className="mt-10">
              <ArchButton href="/#contact">Commission a piece</ArchButton>
            </div>
          </Reveal>
        </div>

        <div className="mt-32">
          <ProductCarousel />
        </div>

        <StarRule className="mx-auto my-24 max-w-3xl" />

        <div className="grid gap-12 lg:grid-cols-12">
          <div className="lg:col-span-3">
            <h3 className="font-serif text-[clamp(1.9rem,3vw,2.6rem)] leading-none">Materials we trust</h3>
            <p className="mt-5 text-ink-muted">Honest, local materials that age well and can always be repaired.</p>
          </div>
          <Reveal as="ul" className="grid grid-cols-2 gap-x-6 gap-y-12 sm:grid-cols-3 lg:col-span-9 lg:grid-cols-5" stagger={0.08}>
            {materials.map((m) => (
              <li key={m.name}>
                <div className="relative aspect-square overflow-hidden rounded-full ring-1 ring-brass/40 ring-offset-4 ring-offset-ivory-deep">
                  <SmartImage src={m.image} alt={`${m.name} sample`} sizes="(min-width: 1024px) 12vw, 40vw" />
                </div>
                <h4 className="mt-5 font-serif text-2xl">{m.name}</h4>
                <p className="eyebrow mt-1 text-[0.62rem] text-terracotta-deep">{m.note}</p>
                <p className="mt-3 text-sm leading-relaxed text-ink-muted">{m.text}</p>
              </li>
            ))}
          </Reveal>
        </div>
      </div>
    </section>
  );
}
