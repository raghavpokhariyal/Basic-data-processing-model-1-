# Placeholder imagery: replace before launch

Every image in this folder is a **computer-generated placeholder render**, not
real project photography. They were modelled and rendered in Three.js (the
chair in the 3D viewer, `lib/three/loungeChair.ts`, is the same model) so the
site could be designed with on-palette, Indian-context imagery.

Replace each file with real photography. The easiest way is to **keep the
file name** and overwrite it. WebP or JPEG at roughly the listed size is
plenty; Next.js resizes and serves AVIF/WebP automatically. If you change
names or formats, update the imports in `lib/images.ts`.

| File | Used for | Suggested size / ratio |
| --- | --- | --- |
| `hero-villa.webp` | Hero slide 1, portfolio, OG image | 1920×1080 (16:9) |
| `hero-living.webp` | Hero slide 2, portfolio | 1920×1080 (16:9) |
| `hero-courtyard.webp` | Hero slide 3, portfolio | 1920×1080 (16:9) |
| `hero-dining.webp` | Hero slide 4, portfolio | 1920×1080 (16:9) |
| `about-courtyard.webp` | About: main arch image | 1100×1375 (4:5) |
| `about-detail.webp` | About: small Mughal-arch image, portfolio | 900×1125 (4:5) |
| `service-construction.webp` | Construction card and page hero | 1100×1375 (4:5) |
| `service-interiors.webp` | Interiors card and page hero, portfolio | 1100×1375 (4:5) |
| `service-furniture.webp` | Furniture card and page hero, 3D viewer poster | 1100×1375 (4:5) |
| `process-consult.webp` … `process-handover.webp` | Process timeline (5 steps) | 900×1125 (4:5) |
| `work-villa-arcade.webp`, `work-site.webp` | Portfolio | 1500×1000 (3:2) |
| `work-jaali-house.webp` | Portfolio | 1100×1375 (4:5) |
| `before-living.webp` / `after-living.webp` | Before/after slider (same camera angle) | 1800×1200 (3:2) |
| `product-*.webp` | Furniture carousel, portfolio | 1000×1250 (4:5) |
| `material-*.webp` | Materials strip | 600×600 (1:1) |

Also replace `app/opengraph-image.jpg` and `app/twitter-image.jpg` (1200×630),
which are built from `hero-villa.webp`.
