import { ArchButton } from '@/components/ui/ArchButton';
import { Mandala } from '@/components/ornaments/Mandala';

export default function NotFound() {
  return (
    <section className="relative grid min-h-[90svh] place-items-center overflow-hidden bg-ivory px-4 pt-28 pb-20 text-center">
      <Mandala className="pointer-events-none absolute top-1/2 left-1/2 h-[46rem] w-[46rem] -translate-x-1/2 -translate-y-1/2 text-brass/15" />
      <div className="relative">
        <p className="eyebrow text-terracotta-deep">Error 404</p>
        <h1 className="display-lg mt-6 font-serif">This room isn’t built yet.</h1>
        <p className="mx-auto mt-6 max-w-md text-lg text-ink-muted">The page you are looking for has moved or never existed. Let us walk you back to the courtyard.</p>
        <div className="mt-10 flex justify-center">
          <ArchButton href="/#top">Return home</ArchButton>
        </div>
      </div>
    </section>
  );
}
