import { cn } from '@/lib/cn';

/** Rings of petals, dots and arcs: faint rangoli line art for backgrounds. */
function petalRing(n: number, r: number, len: number, width: number) {
  const d: string[] = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    const ca = Math.cos(a);
    const sa = Math.sin(a);
    const x0 = 200 + ca * r;
    const y0 = 200 + sa * r;
    const x1 = 200 + ca * (r + len);
    const y1 = 200 + sa * (r + len);
    const nx = -sa * width;
    const ny = ca * width;
    const mx = (x0 + x1) / 2;
    const my = (y0 + y1) / 2;
    d.push(`M${x0.toFixed(2)} ${y0.toFixed(2)}Q${(mx + nx).toFixed(2)} ${(my + ny).toFixed(2)} ${x1.toFixed(2)} ${y1.toFixed(2)}Q${(mx - nx).toFixed(2)} ${(my - ny).toFixed(2)} ${x0.toFixed(2)} ${y0.toFixed(2)}Z`);
  }
  return d.join('');
}

function dotRing(n: number, r: number) {
  return Array.from({ length: n }, (_, i) => {
    const a = (i / n) * Math.PI * 2;
    return { cx: +(200 + Math.cos(a) * r).toFixed(2), cy: +(200 + Math.sin(a) * r).toFixed(2) };
  });
}

function scallops(n: number, r: number, depth: number) {
  let d = '';
  for (let i = 0; i < n; i++) {
    const a0 = (i / n) * Math.PI * 2;
    const a1 = ((i + 1) / n) * Math.PI * 2;
    const am = (a0 + a1) / 2;
    const p0 = [200 + Math.cos(a0) * r, 200 + Math.sin(a0) * r];
    const p1 = [200 + Math.cos(a1) * r, 200 + Math.sin(a1) * r];
    const c = [200 + Math.cos(am) * (r + depth), 200 + Math.sin(am) * (r + depth)];
    d += `${i === 0 ? `M${p0[0].toFixed(2)} ${p0[1].toFixed(2)}` : ''}Q${c[0].toFixed(2)} ${c[1].toFixed(2)} ${p1[0].toFixed(2)} ${p1[1].toFixed(2)}`;
  }
  return d + 'Z';
}

export function Mandala({ className, spin = true }: { className?: string; spin?: boolean }) {
  return (
    <svg viewBox="0 0 400 400" fill="none" stroke="currentColor" strokeWidth="0.8" aria-hidden="true" focusable="false" className={cn(spin && 'mandala-spin', className)}>
      <circle cx="200" cy="200" r="14" />
      <path d={petalRing(8, 16, 34, 9)} />
      <circle cx="200" cy="200" r="56" />
      <path d={petalRing(16, 58, 40, 7)} />
      <path d={scallops(16, 100, 14)} />
      {dotRing(32, 122).map((p, i) => (
        <circle key={i} cx={p.cx} cy={p.cy} r="1.6" fill="currentColor" stroke="none" />
      ))}
      <circle cx="200" cy="200" r="132" />
      <path d={petalRing(24, 134, 36, 6)} />
      <path d={petalRing(24, 140, 22, 3)} transform="rotate(7.5 200 200)" />
      <path d={scallops(48, 172, 8)} />
      <circle cx="200" cy="200" r="190" strokeDasharray="2 5" />
    </svg>
  );
}
