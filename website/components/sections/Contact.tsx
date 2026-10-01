'use client';

import { useRef, useState, type ChangeEvent, type FormEvent } from 'react';
import { SplitReveal, Reveal } from '@/components/motion/Reveal';
import { ArchButton } from '@/components/ui/ArchButton';
import { SocialIcon } from '@/components/ui/SocialIcon';
import { enquiry } from '@/lib/content';
import { validateEnquiry, type EnquiryErrors, type EnquiryInput } from '@/lib/enquiry';
import { mapEmbedSrc, mapLinkHref, site, whatsappHref } from '@/lib/site';
import { cn } from '@/lib/cn';

const empty: EnquiryInput = { name: '', phone: '', email: '', service: '', budget: '', message: '', company: '' };

function Field({
  id,
  label,
  error,
  children,
  className,
}: {
  id: string;
  label: string;
  error?: string;
  children: (props: { id: string; 'aria-invalid': boolean; 'aria-describedby'?: string }) => React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn('field', className)}>
      {children({ id, 'aria-invalid': !!error, 'aria-describedby': error ? `${id}-error` : undefined })}
      <label htmlFor={id} className="field-label">
        {label}
      </label>
      {error && (
        <p id={`${id}-error`} className="mt-2 text-sm text-[#a33a2a]">
          {error}
        </p>
      )}
    </div>
  );
}

export function Contact() {
  const [form, setForm] = useState<EnquiryInput>(empty);
  const [errors, setErrors] = useState<EnquiryErrors>({});
  const [touched, setTouched] = useState<Partial<Record<keyof EnquiryInput, boolean>>>({});
  const [status, setStatus] = useState<'idle' | 'sending' | 'sent' | 'error'>('idle');
  const [serverError, setServerError] = useState('');
  const formRef = useRef<HTMLFormElement>(null);

  const update = (key: keyof EnquiryInput) => (e: ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const next = { ...form, [key]: e.target.value };
    setForm(next);
    if (touched[key]) setErrors(validateEnquiry(next));
  };
  const blur = (key: keyof EnquiryInput) => () => {
    setTouched((t) => ({ ...t, [key]: true }));
    setErrors(validateEnquiry(form));
  };
  const shown = (key: keyof EnquiryInput) => (touched[key] ? errors[key] : undefined);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const v = validateEnquiry(form);
    setErrors(v);
    setTouched({ name: true, phone: true, email: true, service: true, budget: true, message: true });
    if (Object.keys(v).length) {
      const first = Object.keys(v)[0];
      formRef.current?.querySelector<HTMLElement>(`#enquiry-${first}`)?.focus();
      return;
    }
    setStatus('sending');
    setServerError('');
    try {
      const res = await fetch('/api/enquiry', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(form) });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.ok) {
        if (data.errors) setErrors(data.errors);
        throw new Error(data.error ?? 'Something went wrong.');
      }
      setStatus('sent');
      setForm(empty);
    } catch (err) {
      setStatus('error');
      setServerError(err instanceof Error ? err.message : 'Something went wrong.');
    }
  };

  return (
    <section id="contact" aria-labelledby="contact-title" className="bg-blockprint relative bg-sand-light py-28 md:py-40">
      <div className="container-x grid gap-16 lg:grid-cols-12">
        <div className="lg:col-span-5">
          <Reveal>
            <p className="eyebrow flex items-center gap-3 text-terracotta-deep">
              <span className="h-px w-10 bg-current" aria-hidden="true" />
              Begin a project
              <span lang="hi" className="text-[0.95rem] tracking-normal normal-case">
                · संपर्क
              </span>
            </p>
          </Reveal>
          <SplitReveal as="h2" id="contact-title" className="display-lg mt-7 max-w-[12ch] font-serif">
            Let’s build something that belongs to you.
          </SplitReveal>
          <Reveal className="mt-8 space-y-8">
            <p className="max-w-md text-lg leading-relaxed text-ink-muted">
              Tell us a little about your plot, your home or the piece you have in mind. A senior member of the studio will call you within one working day.
            </p>
            <div>
              <ArchButton href={whatsappHref()} external variant="primary" arrow={false} ariaLabel="Chat with us on WhatsApp (opens in a new tab)">
                <span className="flex items-center gap-3">
                  <SocialIcon name="WhatsApp" className="h-5 w-5" />
                  Chat on WhatsApp
                </span>
              </ArchButton>
            </div>
            <dl className="grid gap-6 text-sm sm:grid-cols-2">
              <div>
                <dt className="eyebrow text-[0.62rem] text-terracotta-deep">Studio &amp; workshop</dt>
                <dd className="mt-2 leading-relaxed">
                  {site.address.street}
                  <br />
                  {site.address.locality}, {site.address.region} {site.address.postalCode}
                </dd>
              </div>
              <div>
                <dt className="eyebrow text-[0.62rem] text-terracotta-deep">Hours</dt>
                <dd className="mt-2">{site.hours.display}</dd>
              </div>
              <div>
                <dt className="eyebrow text-[0.62rem] text-terracotta-deep">Call</dt>
                <dd className="mt-2">
                  <a href={`tel:${site.phone.e164}`} className="link-underline">
                    {site.phone.display}
                  </a>
                </dd>
              </div>
              <div>
                <dt className="eyebrow text-[0.62rem] text-terracotta-deep">Write</dt>
                <dd className="mt-2">
                  <a href={`mailto:${site.email}`} className="link-underline">
                    {site.email}
                  </a>
                </dd>
              </div>
            </dl>
            <div className="relative aspect-[4/3] overflow-hidden rounded-t-[999px] border border-brass/40 bg-sand">
              <iframe
                title={`Map showing the ${site.name} studio in ${site.city}`}
                src={mapEmbedSrc}
                loading="lazy"
                referrerPolicy="no-referrer-when-downgrade"
                className="absolute inset-0 h-full w-full [filter:grayscale(0.45)_sepia(0.25)_contrast(0.95)]"
              />
            </div>
            <a href={mapLinkHref} target="_blank" rel="noopener noreferrer" className="eyebrow link-underline inline-block text-[0.66rem]">
              Open in Google Maps ↗
            </a>
          </Reveal>
        </div>

        <div className="lg:col-span-6 lg:col-start-7">
          <div className="relative rounded-t-[min(28vw,16rem)] border border-brass/35 bg-paper px-6 pt-24 pb-10 shadow-[0_40px_80px_-50px_rgba(29,37,57,0.45)] sm:px-12 sm:pt-28 sm:pb-14">
            <p lang="hi" aria-hidden="true" className="absolute inset-x-0 top-10 text-center text-2xl text-terracotta">
              स्वागत है
            </p>
            {status === 'sent' ? (
              <div role="status" className="py-10 text-center">
                <h3 className="font-serif text-4xl">Thank you.</h3>
                <p className="mx-auto mt-4 max-w-sm text-ink-muted">
                  Your enquiry is with us. We will call you within one working day. If it is urgent, message us on WhatsApp.
                </p>
                <div className="mt-8 flex flex-wrap justify-center gap-3">
                  <ArchButton href={whatsappHref()} external variant="outline">
                    WhatsApp us
                  </ArchButton>
                  <ArchButton onClick={() => setStatus('idle')} variant="primary">
                    Send another
                  </ArchButton>
                </div>
              </div>
            ) : (
              <form ref={formRef} onSubmit={submit} noValidate aria-describedby="form-note" className="grid gap-x-8 gap-y-7 sm:grid-cols-2">
                <h3 className="sr-only">Enquiry form</h3>
                <Field id="enquiry-name" label="Your name" error={shown('name')}>
                  {(p) => <input {...p} className="field-input" type="text" name="name" autoComplete="name" placeholder=" " value={form.name} onChange={update('name')} onBlur={blur('name')} required />}
                </Field>
                <Field id="enquiry-phone" label="Mobile number" error={shown('phone')}>
                  {(p) => <input {...p} className="field-input" type="tel" name="phone" inputMode="tel" autoComplete="tel" placeholder=" " value={form.phone} onChange={update('phone')} onBlur={blur('phone')} required />}
                </Field>
                <Field id="enquiry-email" label="Email" error={shown('email')} className="sm:col-span-2">
                  {(p) => <input {...p} className="field-input" type="email" name="email" autoComplete="email" placeholder=" " value={form.email} onChange={update('email')} onBlur={blur('email')} required />}
                </Field>
                <Field id="enquiry-service" label="Service" error={shown('service')}>
                  {(p) => (
                    <select {...p} className="field-input field-select" name="service" value={form.service} onChange={update('service')} onBlur={blur('service')} required>
                      <option value="" disabled>
                        Select a service
                      </option>
                      {enquiry.serviceOptions.map((o) => (
                        <option key={o}>{o}</option>
                      ))}
                    </select>
                  )}
                </Field>
                <Field id="enquiry-budget" label="Budget (₹)" error={shown('budget')}>
                  {(p) => (
                    <select {...p} className="field-input field-select" name="budget" value={form.budget} onChange={update('budget')} onBlur={blur('budget')} required>
                      <option value="" disabled>
                        Select a range
                      </option>
                      {enquiry.budgetOptions.map((o) => (
                        <option key={o}>{o}</option>
                      ))}
                    </select>
                  )}
                </Field>
                <Field id="enquiry-message" label="Tell us about your project" error={shown('message')} className="sm:col-span-2">
                  {(p) => <textarea {...p} className="field-input min-h-32 resize-y" name="message" rows={4} placeholder=" " value={form.message} onChange={update('message')} onBlur={blur('message')} />}
                </Field>
                <div aria-hidden="true" className="absolute -left-[9999px] h-px w-px overflow-hidden">
                  <label htmlFor="enquiry-company">Company</label>
                  <input id="enquiry-company" name="company" tabIndex={-1} autoComplete="off" value={form.company} onChange={update('company')} />
                </div>
                <div className="flex flex-col gap-4 sm:col-span-2 sm:flex-row sm:items-center sm:justify-between">
                  <p id="form-note" className="max-w-xs text-xs leading-relaxed text-ink-muted">
                    We use your details only to respond to this enquiry. No spam, ever.
                  </p>
                  <ArchButton type="submit" disabled={status === 'sending'}>
                    {status === 'sending' ? 'Sending…' : 'Send enquiry'}
                  </ArchButton>
                </div>
                <p role="alert" className="text-sm text-[#a33a2a] sm:col-span-2 empty:hidden">
                  {status === 'error' ? serverError : ''}
                </p>
              </form>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
