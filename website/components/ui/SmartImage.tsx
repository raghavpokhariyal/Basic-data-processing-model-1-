'use client';

import Image, { type StaticImageData } from 'next/image';
import { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/cn';

interface SmartImageProps {
  src: StaticImageData;
  alt: string;
  sizes: string;
  className?: string;
  imgClassName?: string;
  preload?: boolean;
  /** Lazy by default; eager for above-the-fold images that are not preloaded. */
  loading?: 'lazy' | 'eager';
}

/**
 * next/image filling its (sized, positioned) parent, fading from the
 * tiny blurred preview to the sharp image once it has loaded.
 */
export function SmartImage({ src, alt, sizes, className, imgClassName, preload, loading }: SmartImageProps) {
  const [loaded, setLoaded] = useState(false);
  const ref = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const img = ref.current;
    if (img?.complete && img.naturalWidth > 0) setLoaded(true);
  }, []);

  return (
    <div className={cn('absolute inset-0 overflow-hidden', className)}>
      {src.blurDataURL && (
        <div
          aria-hidden="true"
          className="absolute inset-0 scale-110 bg-cover bg-center blur-xl transition-opacity duration-700"
          style={{ backgroundImage: `url(${src.blurDataURL})`, opacity: loaded ? 0 : 1 }}
        />
      )}
      <Image
        ref={ref}
        src={src}
        alt={alt}
        fill
        sizes={sizes}
        preload={preload}
        loading={preload ? undefined : loading}
        className={cn('smart-img object-cover', imgClassName)}
        data-loaded={loaded}
        onLoad={() => setLoaded(true)}
        draggable={false}
      />
    </div>
  );
}
