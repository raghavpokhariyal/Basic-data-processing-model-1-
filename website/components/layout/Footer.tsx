import { TransitionLink } from '@/components/motion/TransitionLink';
import { JaaliDivider } from '@/components/ornaments/Jaali';
import { SocialIcon } from '@/components/ui/SocialIcon';
import { Logo } from '@/components/layout/Header';
import { nav, services } from '@/lib/content';
import { site, whatsappHref } from '@/lib/site';

export function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer data-theme="dark" className="relative overflow-hidden bg-indigo-deep text-sand">
      <JaaliDivider tone="brass" className="opacity-80" />
      <div className="container-x relative pt-20 pb-10">
        <p lang="hi" aria-hidden="true" className="pointer-events-none absolute right-0 bottom-16 font-deva text-[clamp(6rem,18vw,16rem)] leading-none text-ivory/[0.03] select-none">
          {site.nameDevanagari}
        </p>
        <div className="grid gap-14 lg:grid-cols-12">
          <div className="lg:col-span-5">
            <TransitionLink href="/#top" aria-label={`${site.name}, back to top`}>
              <Logo light />
            </TransitionLink>
            <p className="mt-8 max-w-sm font-serif text-3xl leading-tight text-ivory">Built to last. Designed to belong.</p>
            <p className="mt-4 max-w-sm text-sm leading-relaxed">
              Construction, interiors and custom furniture, from foundation to final furnishing. Studio and workshop in {site.city}.
            </p>
          </div>

          <div className="grid gap-10 sm:grid-cols-3 lg:col-span-7">
            <div>
              <h2 className="eyebrow text-brass-light">Studio</h2>
              <address className="mt-5 space-y-2 text-sm not-italic leading-relaxed">
                <p>
                  {site.address.street}
                  <br />
                  {site.address.locality}, {site.address.region} {site.address.postalCode}
                </p>
                <p>
                  <a className="link-underline" href={`tel:${site.phone.e164}`}>
                    {site.phone.display}
                  </a>
                </p>
                <p>
                  <a className="link-underline" href={`mailto:${site.email}`}>
                    {site.email}
                  </a>
                </p>
                <p>{site.hours.display}</p>
              </address>
            </div>
            <div>
              <h2 className="eyebrow text-brass-light">Explore</h2>
              <ul className="mt-5 space-y-2 text-sm">
                {nav.map((n) => (
                  <li key={n.href}>
                    <TransitionLink href={n.href} className="link-underline">
                      {n.label}
                    </TransitionLink>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h2 className="eyebrow text-brass-light">Services</h2>
              <ul className="mt-5 space-y-2 text-sm">
                {services.map((s) => (
                  <li key={s.slug}>
                    <TransitionLink href={`/services/${s.slug}`} className="link-underline">
                      {s.title}
                    </TransitionLink>
                  </li>
                ))}
              </ul>
              <h2 className="eyebrow mt-8 text-brass-light">Follow</h2>
              <ul className="mt-4 flex flex-wrap gap-2">
                {site.social.map((s) => (
                  <li key={s.label}>
                    <a
                      href={s.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      aria-label={`${site.name} on ${s.label}`}
                      className="grid h-10 w-10 place-items-center rounded-full border border-sand/25 transition-colors duration-500 hover:border-brass hover:bg-brass hover:text-indigo-deep"
                    >
                      <SocialIcon name={s.label} />
                    </a>
                  </li>
                ))}
                <li>
                  <a
                    href={whatsappHref()}
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label="Chat on WhatsApp"
                    className="grid h-10 w-10 place-items-center rounded-full border border-sand/25 transition-colors duration-500 hover:border-brass hover:bg-brass hover:text-indigo-deep"
                  >
                    <SocialIcon name="WhatsApp" />
                  </a>
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div className="mt-20 flex flex-col gap-4 border-t border-sand/15 pt-6 pb-16 text-xs text-sand/80 md:flex-row md:items-center md:justify-between md:pr-20 md:pb-0">
          <p>
            © {year} {site.legalName}. All rights reserved.
          </p>
          <p className="tabular-nums">
            GSTIN {site.registration.gstin} <span className="mx-2 opacity-50">·</span> CIN {site.registration.cin}
          </p>
          <TransitionLink href="/#top" className="link-underline self-start md:self-auto">
            Back to top ↑
          </TransitionLink>
        </div>
      </div>
    </footer>
  );
}
