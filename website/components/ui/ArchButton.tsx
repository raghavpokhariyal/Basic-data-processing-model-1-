'use client';

import type { ReactNode } from 'react';
import { Magnetic } from '@/components/motion/Magnetic';
import { TransitionLink } from '@/components/motion/TransitionLink';
import { cn } from '@/lib/cn';

type Variant = 'primary' | 'outline' | 'light' | 'outline-light';

const styles: Record<Variant, { base: string; fill: string; hoverText: string }> = {
  primary: { base: 'bg-terracotta-deep text-ivory', fill: 'bg-indigo', hoverText: '' },
  outline: { base: 'border border-indigo/40 text-indigo', fill: 'bg-indigo', hoverText: 'group-hover:text-ivory group-focus-visible:text-ivory' },
  light: { base: 'bg-ivory text-indigo', fill: 'bg-sand', hoverText: '' },
  'outline-light': { base: 'border border-ivory/50 text-ivory', fill: 'bg-ivory', hoverText: 'group-hover:text-indigo group-focus-visible:text-indigo' },
};

export function Arrow({ className }: { className?: string }) {
  return (
    <svg className={cn('h-3.5 w-3.5 shrink-0', className)} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
      <path d="M1 8h13M9 3l5 5-5 5" />
    </svg>
  );
}

/**
 * Arch-shaped button (rounded head, squared base, like a doorway). On
 * hover a dome of colour rises from the base. Magnetic on fine pointers.
 */
export function ArchButton({
  children,
  href,
  external,
  variant = 'primary',
  className,
  type = 'button',
  onClick,
  disabled,
  arrow = true,
  magnetic = true,
  ariaLabel,
}: {
  children: ReactNode;
  href?: string;
  external?: boolean;
  variant?: Variant;
  className?: string;
  type?: 'button' | 'submit';
  onClick?: () => void;
  disabled?: boolean;
  arrow?: boolean;
  magnetic?: boolean;
  ariaLabel?: string;
}) {
  const s = styles[variant];
  const cls = cn(
    'group relative inline-flex min-h-[3.4rem] items-center justify-center gap-3 overflow-hidden px-8 whitespace-nowrap py-3.5 text-[0.82rem] font-semibold tracking-[0.14em] uppercase transition-colors duration-500',
    'rounded-t-[999px] rounded-b-[0.6rem] disabled:cursor-not-allowed disabled:opacity-60',
    s.base,
    className,
  );
  const inner = (
    <>
      <span
        aria-hidden="true"
        className={cn(
          'absolute inset-x-0 bottom-0 h-[160%] translate-y-full rounded-t-[999px] transition-transform duration-700 ease-out-expo group-hover:translate-y-[18%] group-focus-visible:translate-y-[18%]',
          s.fill,
        )}
      />
      <span className={cn('relative z-10 transition-colors duration-500', s.hoverText)}>{children}</span>
      {arrow && <Arrow className={cn('relative z-10 transition-transform duration-500 group-hover:translate-x-1', s.hoverText)} />}
    </>
  );

  let el: ReactNode;
  if (href && external) {
    const newTab = href.startsWith('http');
    el = (
      <a href={href} className={cls} target={newTab ? '_blank' : undefined} rel={newTab ? 'noopener noreferrer' : undefined} aria-label={ariaLabel}>
        {inner}
      </a>
    );
  } else if (href) {
    el = (
      <TransitionLink href={href} className={cls} aria-label={ariaLabel}>
        {inner}
      </TransitionLink>
    );
  } else {
    el = (
      <button type={type} className={cls} onClick={onClick} disabled={disabled} aria-label={ariaLabel}>
        {inner}
      </button>
    );
  }
  return magnetic ? <Magnetic>{el}</Magnetic> : el;
}
