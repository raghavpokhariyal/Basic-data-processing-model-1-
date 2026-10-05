import type { MetadataRoute } from 'next';
import { site } from '@/lib/site';

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: `${site.name}: Design · Build · Furnish`,
    short_name: site.name,
    description: site.description,
    start_url: '/',
    display: 'standalone',
    background_color: '#f5f0e8',
    theme_color: '#93492f',
    icons: [
      { src: '/icon.svg', sizes: 'any', type: 'image/svg+xml' },
      { src: '/apple-icon.png', sizes: '180x180', type: 'image/png' },
    ],
  };
}
