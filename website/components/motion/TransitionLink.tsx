'use client';

import Link from 'next/link';
import type { ComponentProps, MouseEvent } from 'react';
import { useApp } from '@/components/providers/AppProvider';

type Props = Omit<ComponentProps<typeof Link>, 'href'> & { href: string };

/** next/link that plays the terracotta curtain before moving on. */
export function TransitionLink({ href, onClick, target, ...rest }: Props) {
  const { navigate } = useApp();
  const handle = (e: MouseEvent<HTMLAnchorElement>) => {
    onClick?.(e);
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || target === '_blank') return;
    e.preventDefault();
    navigate(href);
  };
  return <Link href={href} onClick={handle} target={target} scroll={false} {...rest} />;
}
