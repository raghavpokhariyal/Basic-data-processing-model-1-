'use client';

import Image from 'next/image';
import { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { gsap, prefersReducedMotion } from '@/lib/gsap';
import { useApp } from '@/components/providers/AppProvider';
import type { Project } from '@/lib/content';

/** Full-screen project viewer: keyboard, swipe and focus-trapped. */
export function Lightbox({ items, index, onClose, onIndex }: { items: Project[]; index: number; onClose: () => void; onIndex: (i: number) => void }) {
  const { lockScroll } = useApp();
  const root = useRef<HTMLDivElement>(null);
  const closeBtn = useRef<HTMLButtonElement>(null);
  const item = items[index];
  const go = (d: number) => onIndex((index + d + items.length) % items.length);

  // Open: lock scroll, focus, animate in; restore focus on close
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    lockScroll(true);
    closeBtn.current?.focus();
    if (!prefersReducedMotion() && root.current) {
      gsap.fromTo(root.current, { opacity: 0 }, { opacity: 1, duration: 0.5, ease: 'power2.out' });
    }
    return () => {
      lockScroll(false);
      previous?.focus();
    };
  }, [lockScroll]);

  // Image change transition
  useEffect(() => {
    if (prefersReducedMotion()) return;
    const fig = root.current?.querySelector('[data-lb-figure]');
    if (fig) gsap.fromTo(fig, { opacity: 0, scale: 0.97 }, { opacity: 1, scale: 1, duration: 0.7, ease: 'expo.out' });
  }, [index]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      else if (e.key === 'ArrowRight') go(1);
      else if (e.key === 'ArrowLeft') go(-1);
      else if (e.key === 'Tab' && root.current) {
        const f = Array.from(root.current.querySelectorAll<HTMLElement>('button'));
        const i = f.indexOf(document.activeElement as HTMLElement);
        if (e.shiftKey && i <= 0) {
          e.preventDefault();
          f[f.length - 1].focus();
        } else if (!e.shiftKey && i === f.length - 1) {
          e.preventDefault();
          f[0].focus();
        }
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  });

  // Swipe on touch
  const startX = useRef<number | null>(null);

  return createPortal(
    <div
      ref={root}
      role="dialog"
      aria-modal="true"
      aria-label={`${item.title}, ${item.place}`}
      data-theme="dark"
      data-lenis-prevent
      className="fixed inset-0 z-[110] flex flex-col bg-indigo-deep/97 text-ivory"
      onPointerDown={(e) => (startX.current = e.clientX)}
      onPointerUp={(e) => {
        if (startX.current === null) return;
        const dx = e.clientX - startX.current;
        startX.current = null;
        if (Math.abs(dx) > 60) go(dx < 0 ? 1 : -1);
      }}
    >
      <div className="container-x flex h-20 shrink-0 items-center justify-between">
        <p className="eyebrow tabular-nums text-sand">
          {String(index + 1).padStart(2, '0')} / {String(items.length).padStart(2, '0')}
        </p>
        <button ref={closeBtn} type="button" onClick={onClose} className="group flex items-center gap-3 eyebrow text-sand hover:text-ivory" aria-label="Close viewer">
          Close
          <span className="relative grid h-10 w-10 place-items-center rounded-full border border-sand/40 transition-colors group-hover:border-ivory">
            <span className="absolute h-px w-4 rotate-45 bg-current" />
            <span className="absolute h-px w-4 -rotate-45 bg-current" />
          </span>
        </button>
      </div>

      <figure data-lb-figure className="relative flex min-h-0 flex-1 flex-col items-center justify-center px-4 pb-6 sm:px-20">
        <div className="relative h-full w-full">
          <Image key={item.id} src={item.image} alt={item.alt} fill sizes="100vw" className="object-contain" placeholder="blur" />
        </div>
        <figcaption className="mt-5 text-center">
          <p className="font-serif text-3xl">{item.title}</p>
          <p className="eyebrow mt-2 text-sand">
            {item.category} <span className="mx-2 opacity-60">·</span> {item.place} <span className="mx-2 opacity-60">·</span> {item.year}
          </p>
        </figcaption>
      </figure>

      <button type="button" onClick={() => go(-1)} aria-label="Previous project" className="absolute top-1/2 left-2 grid h-14 w-14 -translate-y-1/2 place-items-center rounded-full border border-sand/30 transition-colors hover:bg-ivory hover:text-indigo sm:left-6">
        <svg width="18" height="14" viewBox="0 0 18 14" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
          <path d="M17 7H2M7 1L1 7l6 6" />
        </svg>
      </button>
      <button type="button" onClick={() => go(1)} aria-label="Next project" className="absolute top-1/2 right-2 grid h-14 w-14 -translate-y-1/2 place-items-center rounded-full border border-sand/30 transition-colors hover:bg-ivory hover:text-indigo sm:right-6">
        <svg width="18" height="14" viewBox="0 0 18 14" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
          <path d="M1 7h15M11 1l6 6-6 6" />
        </svg>
      </button>
    </div>,
    document.body,
  );
}
