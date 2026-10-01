import type { SVGProps } from 'react';

/**
 * Brand monogram: the initial drawn inside a pointed Mughal arch, with a
 * small round arch for the crossbar. Paths carry `data-draw` so the
 * preloader can animate them as line drawings.
 */
export function Monogram({ strokeWidth = 2.2, ...props }: SVGProps<SVGSVGElement> & { strokeWidth?: number }) {
  return (
    <svg viewBox="0 0 120 160" fill="none" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false" {...props}>
      <path data-draw pathLength={1} d="M10 152V66C10 40 33 24 60 10C87 24 110 40 110 66V152" />
      <path data-draw pathLength={1} d="M20 152V69C20 47 39 33 60 21C81 33 100 47 100 69V152" strokeWidth={strokeWidth * 0.5} />
      <path data-draw pathLength={1} d="M3 152H117" />
      <path data-draw pathLength={1} d="M38 132L60 56L82 132" />
      <path data-draw pathLength={1} d="M46.5 108C46.5 99.5 52.5 94 60 94C67.5 94 73.5 99.5 73.5 108" />
      <circle data-dot cx="60" cy="5" r="2.6" fill="currentColor" stroke="none" />
    </svg>
  );
}
