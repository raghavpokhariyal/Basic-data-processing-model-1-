import { SplitReveal, Reveal, Counter } from '@/components/motion/Reveal';
import { ArchImage } from '@/components/ui/ArchImage';
import { Mandala } from '@/components/ornaments/Mandala';
import { StarRule } from '@/components/ornaments/Jaali';
import { about } from '@/lib/content';
import { site, yearsOfPractice } from '@/lib/site';

function YearsBadge() {
  const years = yearsOfPractice();
  return (
    <div className="relative grid h-32 w-32 place-items-center rounded-full bg-terracotta-deep text-ivory shadow-[0_20px_50px_-20px_rgba(29,37,57,0.6)] sm:h-36 sm:w-36">
      <svg viewBox="0 0 120 120" className="mandala-spin absolute inset-0 h-full w-full [animation-duration:40s]" aria-hidden="true">
        <defs>
          <path id="badge-circle" d="M60 60m-46 0a46 46 0 1 1 92 0a46 46 0 1 1-92 0" />
        </defs>
        <text className="fill-ivory/85" style={{ fontSize: 9.4, letterSpacing: '0.24em', textTransform: 'uppercase', fontWeight: 600 }}>
          <textPath href="#badge-circle">Years of craft · Since {site.foundingYear} ·</textPath>
        </text>
      </svg>
      <p className="text-center leading-none">
        <span className="block font-serif text-4xl">{years}</span>
        <span className="sr-only"> years of practice, since {site.foundingYear}</span>
      </p>
    </div>
  );
}

export function About() {
  return (
    <section id="about" aria-labelledby="about-title" className="relative overflow-hidden bg-ivory py-28 md:py-40">
      <Mandala className="pointer-events-none absolute -top-24 -right-56 h-[48rem] w-[48rem] text-brass/[0.13]" />

      <div className="container-x relative grid gap-20 lg:grid-cols-12 lg:gap-10">
        <div className="lg:col-span-6 lg:pt-10">
          <Reveal>
            <p className="eyebrow flex items-center gap-3 text-terracotta-deep">
              <span className="h-px w-10 bg-current" aria-hidden="true" />
              {about.eyebrow}
              <span lang="hi" className="text-[0.95rem] tracking-normal normal-case">
                · {about.eyebrowHi}
              </span>
            </p>
          </Reveal>
          <SplitReveal as="h2" id="about-title" className="display-lg mt-7 max-w-[13ch] font-serif">
            {about.heading}
          </SplitReveal>
          <Reveal className="mt-10 max-w-xl space-y-6 text-lg leading-relaxed text-ink-muted">
            {about.paragraphs.map((p) => (
              <p key={p.slice(0, 24)}>{p}</p>
            ))}
          </Reveal>
          <Reveal>
            <figure className="mt-14 max-w-xl border-l border-brass/70 pl-7">
              <blockquote className="font-serif text-[1.7rem] leading-snug text-indigo italic">“{about.quote}”</blockquote>
              <figcaption className="mt-5 text-sm text-ink-muted">
                <span className="font-semibold text-indigo">{site.founder.name}</span>, {site.founder.role}
              </figcaption>
            </figure>
          </Reveal>
        </div>

        <div className="relative mx-auto w-full max-w-md lg:col-span-5 lg:col-start-8 lg:max-w-none">
          <div className="relative">
            <div aria-hidden="true" className="arch absolute inset-0 translate-x-5 translate-y-5 border border-brass/60" />
            <ArchImage
              src={about.images.main.src}
              alt={about.images.main.alt}
              sizes="(min-width: 1024px) 38vw, (min-width: 640px) 28rem, 90vw"
              shape="round"
              reveal="wipe"
              className="aspect-[4/5] w-full"
            />
          </div>
          <div className="absolute -bottom-14 -left-4 w-[46%] sm:-left-12 lg:-left-24">
            <ArchImage
              src={about.images.detail.src}
              alt={about.images.detail.alt}
              sizes="(min-width: 1024px) 18vw, 45vw"
              shape="mughal"
              reveal="jaali"
              tone="ivory"
              className="aspect-[4/5] w-full shadow-[0_30px_60px_-30px_rgba(29,37,57,0.55)]"
            />
          </div>
          <div className="absolute -top-10 -right-2 sm:-right-8">
            <YearsBadge />
          </div>
        </div>
      </div>

      <div className="container-x relative mt-36">
        <StarRule className="mx-auto max-w-5xl" />
        <dl className="mx-auto mt-14 grid max-w-6xl grid-cols-2 gap-y-12 md:grid-cols-4">
          {about.stats.map((s, i) => (
            <div key={s.label} className={`flex flex-col-reverse items-center text-center ${i > 0 ? 'md:border-l md:border-brass/30' : ''}`}>
              <dt className="eyebrow mt-3 text-ink-muted">{s.label}</dt>
              <dd className="font-serif text-[clamp(2.4rem,5vw,4.25rem)] leading-none text-indigo">
                <Counter value={s.value} suffix={s.suffix} />
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
