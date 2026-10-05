'use client';

import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { SplitText } from 'gsap/SplitText';
import { Flip } from 'gsap/Flip';
import { useGSAP } from '@gsap/react';

if (typeof window !== 'undefined') {
  gsap.registerPlugin(ScrollTrigger, SplitText, Flip, useGSAP);
  gsap.defaults({ ease: 'power3.out', duration: 1 });
  ScrollTrigger.config({ ignoreMobileResize: true });
}

/** Media conditions shared by every animated component. */
export const MQ = {
  motion: '(prefers-reduced-motion: no-preference)',
  reduce: '(prefers-reduced-motion: reduce)',
  desktop: '(min-width: 768px)',
  fine: '(hover: hover) and (pointer: fine)',
} as const;

export const prefersReducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia(MQ.reduce).matches;

export { gsap, ScrollTrigger, SplitText, Flip, useGSAP };
