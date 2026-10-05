'use client';

import { useEffect, useRef, useState } from 'react';
import { gsap, MQ } from '@/lib/gsap';

/**
 * Custom cursor: a brass dot with a trailing ring. Over elements carrying
 * `data-cursor="View"` (or Drag, Explore, Rotate…) the ring grows into a
 * terracotta disc with that label. Only on fine pointers without
 * reduced-motion; the native cursor is restored otherwise.
 */
export function Cursor() {
  const dot = useRef<HTMLDivElement>(null);
  const ring = useRef<HTMLDivElement>(null);
  const [label, setLabel] = useState('');
  const [enabled, setEnabled] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia(`${MQ.fine} and ${MQ.motion}`);
    const update = () => setEnabled(mq.matches);
    update();
    mq.addEventListener('change', update);
    return () => mq.removeEventListener('change', update);
  }, []);

  useEffect(() => {
    if (!enabled || !dot.current || !ring.current) return;
    const html = document.documentElement;
    html.classList.add('has-cursor');
    const d = dot.current;
    const r = ring.current;
    gsap.set([d, r], { xPercent: -50, yPercent: -50, x: -100, y: -100 });
    const dx = gsap.quickTo(d, 'x', { duration: 0.12, ease: 'power3' });
    const dy = gsap.quickTo(d, 'y', { duration: 0.12, ease: 'power3' });
    const rx = gsap.quickTo(r, 'x', { duration: 0.5, ease: 'power3' });
    const ry = gsap.quickTo(r, 'y', { duration: 0.5, ease: 'power3' });
    let visible = false;
    let mode: 'idle' | 'link' | 'label' = 'idle';

    const setMode = (next: typeof mode, text = '') => {
      if (next === mode && text === r.dataset.text) return;
      mode = next;
      r.dataset.text = text;
      setLabel(text);
      gsap.to(r, {
        width: next === 'label' ? 92 : next === 'link' ? 54 : 34,
        height: next === 'label' ? 92 : next === 'link' ? 54 : 34,
        backgroundColor: next === 'label' ? 'rgba(147,73,47,0.94)' : 'rgba(147,73,47,0)',
        borderColor: next === 'label' ? 'rgba(147,73,47,0)' : 'rgba(176,141,87,0.75)',
        duration: 0.45,
        ease: 'expo.out',
      });
      gsap.to(d, { scale: next === 'idle' ? 1 : 0, duration: 0.3 });
    };

    const move = (e: PointerEvent) => {
      if (e.pointerType !== 'mouse') return;
      if (!visible) {
        visible = true;
        gsap.set([d, r], { x: e.clientX, y: e.clientY });
        gsap.to([d, r], { opacity: 1, duration: 0.3 });
      }
      dx(e.clientX);
      dy(e.clientY);
      rx(e.clientX);
      ry(e.clientY);
    };
    const over = (e: PointerEvent) => {
      const t = e.target as Element | null;
      const labelled = t?.closest<HTMLElement>('[data-cursor]');
      if (labelled) return setMode('label', labelled.dataset.cursor ?? '');
      if (t?.closest('a, button, [role="button"], label, select, summary, input[type="range"]')) return setMode('link');
      setMode('idle');
    };
    const leave = () => {
      visible = false;
      gsap.to([d, r], { opacity: 0, duration: 0.3 });
    };
    const down = () => gsap.to(r, { scale: 0.85, duration: 0.2 });
    const up = () => gsap.to(r, { scale: 1, duration: 0.3 });

    window.addEventListener('pointermove', move, { passive: true });
    document.addEventListener('pointerover', over, { passive: true });
    document.documentElement.addEventListener('pointerleave', leave);
    window.addEventListener('pointerdown', down);
    window.addEventListener('pointerup', up);
    return () => {
      html.classList.remove('has-cursor');
      window.removeEventListener('pointermove', move);
      document.removeEventListener('pointerover', over);
      document.documentElement.removeEventListener('pointerleave', leave);
      window.removeEventListener('pointerdown', down);
      window.removeEventListener('pointerup', up);
    };
  }, [enabled]);

  if (!enabled) return null;
  return (
    <div aria-hidden="true" className="pointer-events-none fixed inset-0 z-[120]">
      <div ref={ring} className="absolute top-0 left-0 grid h-[34px] w-[34px] place-items-center rounded-full border opacity-0" style={{ borderColor: 'rgba(176,141,87,0.75)' }}>
        <span className="eyebrow text-[0.62rem] tracking-[0.18em] text-ivory">{label}</span>
      </div>
      <div ref={dot} className="absolute top-0 left-0 h-[6px] w-[6px] rounded-full bg-terracotta-deep opacity-0" />
    </div>
  );
}
