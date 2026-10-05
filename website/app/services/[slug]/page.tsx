import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { PageHero } from '@/components/sections/PageHero';
import { SplitReveal, Reveal } from '@/components/motion/Reveal';
import { TransitionLink } from '@/components/motion/TransitionLink';
import { ArchImage } from '@/components/ui/ArchImage';
import { ArchButton, Arrow } from '@/components/ui/ArchButton';
import { SmartImage } from '@/components/ui/SmartImage';
import { Mandala } from '@/components/ornaments/Mandala';
import { JaaliDivider } from '@/components/ornaments/Jaali';
import { JsonLd } from '@/components/seo/JsonLd';
import { projects, serviceBySlug, services } from '@/lib/content';
import { site } from '@/lib/site';

type Params = { params: Promise<{ slug: string }> };

export const dynamicParams = false;

export function generateStaticParams() {
  return services.map((s) => ({ slug: s.slug }));
}

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { slug } = await params;
  const s = serviceBySlug(slug);
  if (!s) return {};
  const title = `${s.title} in ${site.city}`;
  return {
    title,
    description: `${s.tagline} ${s.summary}`,
    alternates: { canonical: `/services/${s.slug}` },
    openGraph: { title: `${title} | ${site.name}`, description: s.summary, url: `/services/${s.slug}`, images: [{ url: s.image.src, width: s.image.width, height: s.image.height, alt: s.imageAlt }] },
  };
}

export default async function ServicePage({ params }: Params) {
  const { slug } = await params;
  const s = serviceBySlug(slug);
  if (!s) notFound();
  const idx = services.indexOf(s);
  const next = services[(idx + 1) % services.length];
  const gallery = projects.filter((p) => p.category === s.category && p.image !== s.image).slice(0, 3);

  const schema = [
    {
      '@context': 'https://schema.org',
      '@type': 'Service',
      name: s.title,
      description: s.summary,
      serviceType: s.title,
      provider: { '@id': `${site.url}/#business` },
      areaServed: { '@type': 'City', name: site.city },
      url: `${site.url}/services/${s.slug}`,
    },
    {
      '@context': 'https://schema.org',
      '@type': 'BreadcrumbList',
      itemListElement: [
        { '@type': 'ListItem', position: 1, name: 'Home', item: site.url },
        { '@type': 'ListItem', position: 2, name: 'Services', item: `${site.url}/#services` },
        { '@type': 'ListItem', position: 3, name: s.title, item: `${site.url}/services/${s.slug}` },
      ],
    },
    {
      '@context': 'https://schema.org',
      '@type': 'FAQPage',
      mainEntity: s.faqs.map((f) => ({ '@type': 'Question', name: f.q, acceptedAnswer: { '@type': 'Answer', text: f.a } })),
    },
  ];

  return (
    <>
      <JsonLd data={schema} />
      <PageHero
        eyebrow={`${s.number} · ${s.transliteration}`}
        title={s.title}
        hindi={s.hindi}
        lead={s.tagline + ' ' + s.summary}
        image={s.image}
        alt={s.imageAlt}
        crumbs={[{ label: 'Home', href: '/#top' }, { label: 'Services', href: '/#services' }, { label: s.title }]}
      />

      {/* Intro + highlights */}
      <section aria-labelledby="intro-title" className="bg-ivory py-24 md:py-36">
        <div className="container-x grid gap-16 lg:grid-cols-12">
          <div className="lg:col-span-7">
            <p className="eyebrow text-terracotta-deep">Our approach</p>
            <SplitReveal as="h2" id="intro-title" className="mt-6 font-serif text-[clamp(2rem,3.6vw,3.25rem)] leading-[1.08]">
              {s.intro}
            </SplitReveal>
          </div>
          <Reveal as="dl" className="grid content-start gap-8 border-l border-brass/40 pl-8 lg:col-span-4 lg:col-start-9">
            {s.highlights.map((h) => (
              <div key={h.label} className="flex flex-col-reverse">
                <dt className="eyebrow mt-2 text-ink-muted">{h.label}</dt>
                <dd className="font-serif text-5xl leading-none">{h.value}</dd>
              </div>
            ))}
          </Reveal>
        </div>
      </section>

      {/* What's included */}
      <section aria-labelledby="included-title" className="bg-blockprint bg-sand-light">
        <JaaliDivider tone="terracotta" className="opacity-60" />
        <div className="container-x py-24 md:py-32">
          <h2 id="included-title" className="display-md max-w-[16ch] font-serif">
            What is included
          </h2>
          <Reveal as="ol" className="mt-14 grid gap-px overflow-hidden rounded-t-[3rem] border border-brass/30 bg-brass/30 sm:grid-cols-2 lg:grid-cols-3" stagger={0.07}>
            {s.included.map((item, i) => (
              <li key={item.title} className="bg-paper p-8 sm:p-10">
                <p className="eyebrow text-terracotta-deep">0{i + 1}</p>
                <h3 className="mt-4 font-serif text-3xl leading-tight">{item.title}</h3>
                <p className="mt-3 leading-relaxed text-ink-muted">{item.text}</p>
              </li>
            ))}
          </Reveal>
        </div>
      </section>

      {/* Gallery */}
      {gallery.length > 0 && (
        <section aria-labelledby="gallery-title" className="bg-ivory py-24 md:py-36">
          <div className="container-x">
            <div className="flex flex-wrap items-end justify-between gap-6">
              <h2 id="gallery-title" className="display-md font-serif">
                Recent {s.title.toLowerCase()} work
              </h2>
              <TransitionLink href="/#work" className="eyebrow link-underline inline-flex items-center gap-2 text-terracotta-deep">
                All projects <Arrow />
              </TransitionLink>
            </div>
            <ul className="mt-14 grid gap-10 md:grid-cols-3 md:gap-8">
              {gallery.map((p, i) => (
                <li key={p.id} className={i === 1 ? 'md:mt-20' : ''}>
                  <ArchImage src={p.image} alt={p.alt} sizes="(min-width: 768px) 31vw, 92vw" shape={i === 1 ? 'mughal' : 'round'} reveal={i === 1 ? 'jaali' : 'wipe'} className="aspect-[4/5] w-full" />
                  <p className="mt-5 font-serif text-2xl">{p.title}</p>
                  <p className="eyebrow mt-1 text-ink-muted">
                    {p.place} · {p.year}
                  </p>
                </li>
              ))}
            </ul>
          </div>
        </section>
      )}

      {/* FAQ */}
      <section aria-labelledby="faq-title" className="border-t border-brass/25 bg-ivory py-24 md:py-32">
        <div className="container-x grid gap-12 lg:grid-cols-12">
          <h2 id="faq-title" className="display-md font-serif lg:col-span-4">
            Questions, answered
          </h2>
          <div className="divide-y divide-brass/30 border-y border-brass/30 lg:col-span-7 lg:col-start-6">
            {s.faqs.map((f) => (
              <details key={f.q} className="group py-6">
                <summary className="flex cursor-pointer list-none items-center justify-between gap-6 font-serif text-2xl leading-snug [&::-webkit-details-marker]:hidden">
                  {f.q}
                  <span aria-hidden="true" className="relative grid h-9 w-9 shrink-0 place-items-center rounded-full border border-indigo/25 transition-transform duration-500 group-open:rotate-45">
                    <span className="absolute h-px w-3.5 bg-current" />
                    <span className="absolute h-3.5 w-px bg-current" />
                  </span>
                </summary>
                <p className="mt-4 max-w-2xl leading-relaxed text-ink-muted">{f.a}</p>
              </details>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section data-theme="dark" className="relative overflow-hidden bg-indigo-deep py-28 text-ivory md:py-36">
        <Mandala className="pointer-events-none absolute top-1/2 left-1/2 h-[56rem] w-[56rem] -translate-x-1/2 -translate-y-1/2 text-brass/[0.14]" />
        <div className="container-x relative text-center">
          <p lang="hi" aria-hidden="true" className="text-2xl text-terracotta-light">
            {s.hindi}
          </p>
          <h2 className="display-lg mx-auto mt-4 max-w-[16ch] font-serif">Ready when you are.</h2>
          <p className="mx-auto mt-6 max-w-lg text-lg text-sand">Share a few details and we will set up a conversation, at your site or in our studio.</p>
          <div className="mt-10 flex flex-wrap justify-center gap-3">
            <ArchButton href="/#contact" variant="light">
              Begin your project
            </ArchButton>
            <ArchButton href={`tel:${site.phone.e164}`} external variant="outline-light" arrow={false}>
              Call {site.phone.display}
            </ArchButton>
          </div>
        </div>
      </section>

      {/* Next service */}
      <TransitionLink href={`/services/${next.slug}`} data-cursor="Next" className="group relative block overflow-hidden bg-indigo text-ivory">
        <div className="absolute inset-0 opacity-40 transition-opacity duration-1000 group-hover:opacity-70">
          <SmartImage src={next.image} alt="" sizes="100vw" imgClassName="transition-transform duration-[1800ms] ease-out-expo group-hover:scale-105" />
        </div>
        <div aria-hidden="true" className="absolute inset-0 bg-gradient-to-r from-indigo-deep/90 to-indigo-deep/30" />
        <div className="container-x relative flex items-end justify-between gap-6 py-20 md:py-28">
          <div>
            <p className="eyebrow text-sand">Next service · {next.number}</p>
            <p className="display-lg mt-4 font-serif">{next.title}</p>
          </div>
          <span className="grid h-16 w-16 shrink-0 place-items-center rounded-full border border-ivory/50 transition-colors duration-500 group-hover:bg-ivory group-hover:text-indigo">
            <Arrow className="h-5 w-5" />
          </span>
        </div>
      </TransitionLink>
    </>
  );
}
