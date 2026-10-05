'use client';

import { useEffect, useState } from 'react';
import { SocialIcon } from '@/components/ui/SocialIcon';
import { whatsappHref } from '@/lib/site';
import { cn } from '@/lib/cn';

/** Floating click-to-chat button that appears once the visitor scrolls. */
export function WhatsAppFab() {
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const onScroll = () => setShown(window.scrollY > window.innerHeight * 0.8);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);
  return (
    <a
      href={whatsappHref()}
      target="_blank"
      rel="noopener noreferrer"
      aria-label="Chat with us on WhatsApp"
      tabIndex={shown ? 0 : -1}
      className={cn(
        'group fixed right-4 bottom-4 z-[70] flex h-14 items-center gap-0 overflow-hidden rounded-t-[999px] rounded-b-2xl bg-terracotta-deep pr-4 pl-4 text-ivory shadow-[0_18px_40px_-18px_rgba(29,37,57,0.6)] transition-all duration-700 ease-out-expo sm:right-6 sm:bottom-6',
        shown ? 'translate-y-0 opacity-100' : 'pointer-events-none translate-y-6 opacity-0',
      )}
    >
      <SocialIcon name="WhatsApp" className="h-6 w-6 shrink-0" />
      <span className="max-w-0 overflow-hidden text-xs font-semibold tracking-[0.12em] whitespace-nowrap uppercase transition-[max-width,margin] duration-700 ease-out-expo group-hover:ml-3 group-hover:max-w-40 group-focus-visible:ml-3 group-focus-visible:max-w-40">
        Chat with us
      </span>
    </a>
  );
}
