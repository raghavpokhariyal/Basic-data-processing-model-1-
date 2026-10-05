'use client';

import { usePathname } from 'next/navigation';
import { useCallback, useEffect, useRef, useState } from 'react';
import { gsap, prefersReducedMotion } from '@/lib/gsap';
import { useApp } from '@/components/providers/AppProvider';
import { TransitionLink } from '@/components/motion/TransitionLink';
import { Monogram } from '@/components/ornaments/Monogram';
import { Mandala } from '@/components/ornaments/Mandala';
import { ArchButton } from '@/components/ui/ArchButton';
import { nav } from '@/lib/content';
import { site, whatsappHref } from '@/lib/site';
import { cn } from '@/lib/cn';

const hindiFor: Record<string, string> = {
  About: 'परिचय',
  Services: 'सेवाएँ',
  Process: 'प्रक्रिया',
  Work: 'कार्य',
  Furniture: 'फ़र्नीचर',
  Contact: 'संपर्क',
};

export function Logo({ light }: { light?: boolean }) {
  return (
    <span className={cn('flex items-center gap-3', light ? 'text-ivory' : 'text-indigo')}>
      <Monogram className={cn('h-10 w-auto', light ? 'text-ivory' : 'text-terracotta-deep')} strokeWidth={3} />
      <span className="flex flex-col leading-none">
        <span className="font-serif text-[1.45rem] font-semibold tracking-[0.28em] uppercase">{site.name}</span>
        <span className="eyebrow mt-1 text-[0.55rem] tracking-[0.3em] opacity-75">{site.tagline}</span>
      </span>
    </span>
  );
}

export function Header() {
  const pathname = usePathname();
  const { navigate, lockScroll } = useApp();
  const [overHero, setOverHero] = useState(true);
  const [hidden, setHidden] = useState(false);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState<string>('');
  const menuRef = useRef<HTMLDivElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);

  // Transparent over a dark hero, solid ivory elsewhere
  useEffect(() => {
    const hero = document.querySelector('[data-hero]');
    if (!hero) {
      setOverHero(false);
      return;
    }
    const io = new IntersectionObserver(([e]) => setOverHero(e.isIntersecting), { rootMargin: '-88px 0px 0px 0px', threshold: 0 });
    io.observe(hero);
    return () => io.disconnect();
  }, [pathname]);

  // Hide on scroll down, reveal on scroll up
  useEffect(() => {
    let last = window.scrollY;
    const onScroll = () => {
      const y = window.scrollY;
      if (Math.abs(y - last) < 6) return;
      setHidden(y > last && y > 240);
      last = y;
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  // Highlight the section in view
  useEffect(() => {
    const ids = nav.map((n) => n.href.split('#')[1]);
    const sections = ids.map((id) => document.getElementById(id)).filter(Boolean) as HTMLElement[];
    if (!sections.length) {
      setActive('');
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) if (e.isIntersecting) setActive(e.target.id);
      },
      { rootMargin: '-45% 0px -50% 0px' },
    );
    sections.forEach((s) => io.observe(s));
    return () => io.disconnect();
  }, [pathname]);

  const close = useCallback(() => setOpen(false), []);

  // Menu open/close animation, scroll lock, focus management
  useEffect(() => {
    const menu = menuRef.current;
    if (!menu) return;
    const reduce = prefersReducedMotion();
    if (open) {
      lockScroll(true);
      gsap.set(menu, { display: 'flex' });
      const tl = gsap.timeline();
      tl.fromTo(menu, { clipPath: 'inset(0% 0% 100% 0%)' }, { clipPath: 'inset(0% 0% 0% 0%)', duration: reduce ? 0 : 0.8, ease: 'expo.inOut' }).fromTo(
        menu.querySelectorAll('[data-menu-item]'),
        { yPercent: reduce ? 0 : 110, opacity: reduce ? 1 : 0 },
        { yPercent: 0, opacity: 1, duration: reduce ? 0 : 0.9, stagger: 0.06, ease: 'expo.out' },
        '-=0.35',
      );
      const first = menu.querySelector<HTMLElement>('a, button');
      first?.focus();
      const onKey = (e: KeyboardEvent) => {
        if (e.key === 'Escape') setOpen(false);
        if (e.key === 'Tab') {
          const items = Array.from(menu.querySelectorAll<HTMLElement>('a, button'));
          const i = items.indexOf(document.activeElement as HTMLElement);
          if (e.shiftKey && i <= 0) {
            e.preventDefault();
            items[items.length - 1].focus();
          } else if (!e.shiftKey && i === items.length - 1) {
            e.preventDefault();
            items[0].focus();
          }
        }
      };
      document.addEventListener('keydown', onKey);
      return () => document.removeEventListener('keydown', onKey);
    }
    if (menu.style.display === 'flex') {
      gsap.to(menu, {
        clipPath: 'inset(0% 0% 100% 0%)',
        duration: reduce ? 0 : 0.6,
        ease: 'expo.inOut',
        onComplete: () => {
          gsap.set(menu, { display: 'none' });
        },
      });
      lockScroll(false);
      toggleRef.current?.focus();
    }
  }, [open, lockScroll]);

  const light = overHero && !open;

  return (
    <>
      <header
        className={cn(
          'fixed inset-x-0 top-0 z-[80] transition-[transform,background-color,box-shadow,color] duration-700 ease-out-expo',
          hidden && !open ? '-translate-y-full' : 'translate-y-0',
          light ? 'bg-transparent text-ivory' : 'bg-ivory/88 text-indigo shadow-[0_1px_0_rgba(176,141,87,0.28)] backdrop-blur-md',
        )}
      >
        {light && <div aria-hidden="true" className="pointer-events-none absolute inset-0 -bottom-10 bg-gradient-to-b from-indigo-deep/55 to-transparent" />}
        <div className="container-x relative flex h-[4.75rem] items-center justify-between gap-6">
          <TransitionLink href="/#top" className="shrink-0" aria-label={`${site.name}, home`}>
            <Logo light={light} />
          </TransitionLink>
          <nav aria-label="Primary" className="hidden lg:block">
            <ul className="flex items-center gap-9">
              {nav.map((item) => {
                const id = item.href.split('#')[1];
                return (
                  <li key={item.href}>
                    <TransitionLink
                      href={item.href}
                      aria-current={active === id ? 'true' : undefined}
                      className="link-underline pb-1 text-[0.8rem] font-semibold tracking-[0.14em] uppercase"
                    >
                      {item.label}
                    </TransitionLink>
                  </li>
                );
              })}
            </ul>
          </nav>
          <div className="flex items-center gap-3">
            <div className="hidden sm:block">
              <ArchButton href="/#contact" variant={light ? 'light' : 'primary'} className="min-h-[2.9rem] px-6 text-[0.72rem]">
                Enquire
              </ArchButton>
            </div>
            <button
              ref={toggleRef}
              type="button"
              className="relative grid h-11 w-11 place-items-center lg:hidden"
              aria-expanded={open}
              aria-controls="mobile-menu"
              aria-label={open ? 'Close menu' : 'Open menu'}
              onClick={() => setOpen((v) => !v)}
            >
              <span className={cn('absolute h-px w-7 bg-current transition-transform duration-500', open ? 'rotate-45' : '-translate-y-1.5')} />
              <span className={cn('absolute h-px w-7 bg-current transition-transform duration-500', open ? '-rotate-45' : 'translate-y-1.5')} />
            </button>
          </div>
        </div>
      </header>

      <div
        ref={menuRef}
        id="mobile-menu"
        role="dialog"
        aria-modal="true"
        aria-label="Menu"
        data-theme="dark"
        className="fixed inset-0 z-[79] flex-col overflow-y-auto bg-indigo-deep pt-24 text-ivory lg:hidden"
        style={{ display: 'none' }}
      >
        <Mandala className="pointer-events-none absolute -right-40 -bottom-40 h-[34rem] w-[34rem] text-brass/15" />
        <nav aria-label="Mobile" className="container-x relative flex-1">
          <ul className="space-y-1">
            {nav.map((item, i) => (
              <li key={item.href} className="overflow-hidden">
                <a
                  data-menu-item
                  href={item.href}
                  onClick={(e) => {
                    e.preventDefault();
                    close();
                    setTimeout(() => navigate(item.href), 250);
                  }}
                  className="flex items-baseline gap-4 py-2"
                >
                  <span className="eyebrow w-6 text-brass-light">0{i + 1}</span>
                  <span className="font-serif text-5xl leading-tight">{item.label}</span>
                  <span lang="hi" aria-hidden="true" className="text-base text-sand/70">
                    {hindiFor[item.label]}
                  </span>
                </a>
              </li>
            ))}
          </ul>
        </nav>
        <div className="container-x relative space-y-3 py-10 text-sm text-sand">
          <p data-menu-item>{site.address.street}, {site.address.locality}</p>
          <p data-menu-item>
            <a href={`tel:${site.phone.e164}`} className="link-underline">{site.phone.display}</a>
            <span className="mx-2 opacity-50">·</span>
            <a href={whatsappHref()} target="_blank" rel="noopener noreferrer" className="link-underline">WhatsApp</a>
          </p>
        </div>
      </div>
    </>
  );
}
