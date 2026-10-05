import { cn } from '@/lib/cn';

const TONES = {
  brass: 'var(--color-brass)',
  ivory: 'var(--color-ivory)',
  terracotta: 'var(--color-terracotta)',
} as const;

/**
 * A strip of Mughal jaali (eight-point stars and diamonds) used as a
 * section divider and as the footer border.
 */
export function JaaliDivider({ className, tone = 'brass' }: { className?: string; tone?: keyof typeof TONES }) {
  const color = TONES[tone];
  const id = `jaali-strip-${tone}`;
  return (
    <div className={cn('border-y', className)} style={{ borderColor: `color-mix(in oklab, ${color} 60%, transparent)` }} aria-hidden="true">
      <svg className="block h-7 w-full" focusable="false">
        <defs>
          <pattern id={id} width="28" height="28" patternUnits="userSpaceOnUse">
            <g fill="none" stroke={color} strokeWidth="1">
              <path d="M14 3l3.2 7.8L25 14l-7.8 3.2L14 25l-3.2-7.8L3 14l7.8-3.2z" />
              <rect x="8.5" y="8.5" width="11" height="11" transform="rotate(45 14 14)" />
              <path d="M0 0l3 3M28 0l-3 3M0 28l3-3M28 28l-3-3" />
            </g>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill={`url(#${id})`} opacity="0.85" />
      </svg>
    </div>
  );
}

/** Small ornamental rule: hairline, a jaali star, hairline. */
export function StarRule({ className }: { className?: string }) {
  return (
    <div className={cn('flex items-center gap-3 text-brass', className)} aria-hidden="true">
      <span className="h-px flex-1 bg-current opacity-50" />
      <svg width="18" height="18" viewBox="0 0 28 28" fill="none" stroke="currentColor" strokeWidth="1.4">
        <path d="M14 3l3.2 7.8L25 14l-7.8 3.2L14 25l-3.2-7.8L3 14l7.8-3.2z" />
        <circle cx="14" cy="14" r="2" fill="currentColor" stroke="none" />
      </svg>
      <span className="h-px flex-1 bg-current opacity-50" />
    </div>
  );
}
