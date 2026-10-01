# Aangan: design-and-build studio website

A premium, fully responsive site for an Indian design-and-build company
working across **Construction**, **Interior Design** and **Custom Furniture**.
The look is modern Indian luxury: an ivory, sandstone, terracotta, indigo and
brass palette, jaali lattices, Mughal and Rajasthani arches, faint rangoli
line art, a block-print ground and a paper-grain overlay.

> **Brand placeholders.** The brief left the name and city as `[BRAND NAME]`
> and `[CITY]`. The site uses the stand-in **“Aangan”** (आँगन, courtyard) and
> **Jaipur**. Change both, plus the address, phone, email and GST/CIN, in one
> file: [`lib/site.ts`](lib/site.ts).

## Stack

- **Next.js 16** (App Router, React 19, TypeScript), static pages plus one API route
- **Tailwind CSS 4** (design tokens in `app/globals.css`)
- **GSAP 3** with ScrollTrigger, SplitText and Flip, plus **Lenis** smooth scrolling
- **Three.js** for the interactive furniture viewer, loaded only when it nears the viewport
- `next/font` (Cormorant Garamond, Manrope, Tiro Devanagari Hindi) and `next/image` (AVIF/WebP, blur previews)

## Getting started

```bash
cd website
npm install
npm run dev          # http://localhost:3000
npm run build && npm start
npm run typecheck
```

Copy `.env.example` to `.env.local` and set:

| Variable | Purpose |
| --- | --- |
| `NEXT_PUBLIC_SITE_URL` | Public URL, used for canonical links, the sitemap, Open Graph and JSON-LD |
| `ENQUIRY_WEBHOOK_URL` | Where contact-form enquiries are POSTed as JSON (Zapier, Make, Slack, a CRM…). Without it they are only logged on the server. |

## What to replace before launch

1. **Brand and contact details:** `lib/site.ts` (name, city, address, map query, phone, WhatsApp number, email, hours, founder, GSTIN, CIN, social links).
2. **Copy, projects, products and testimonials:** `lib/content.ts`. Project names and testimonials are illustrative and are marked `PLACEHOLDER`; use real client quotes only with permission.
3. **Images:** every image is a clearly labelled placeholder render. See [`assets/placeholders/README.md`](assets/placeholders/README.md) for each file, where it appears and the suggested size. Overwrite files with the same name, or edit `lib/images.ts`.
4. **Social preview and icons:** `app/opengraph-image.jpg`, `app/twitter-image.jpg`, `app/icon.svg`, `app/apple-icon.png`.
5. **Monogram:** `components/ornaments/Monogram.tsx` draws the initial inside an arch. Adjust the letter paths if the brand initial changes.

## Page map

| Section | Highlights |
| --- | --- |
| Preloader | Monogram drawn as SVG line art inside a Mughal arch, then a terracotta curtain wipe. Repeat visits in a session get only the wipe. |
| Hero | Four-slide slideshow with clip-path transitions and Ken Burns, word-by-word headline reveal, parallax, progress bar and an arch-shaped scroll cue |
| About | Story, founder quote, arch and jaali image reveals, a rotating “years of craft” badge, animated counters (Indian digit grouping: “12 Lakh+”) |
| Services | Three Mughal-arch cards with zoom-on-hover, slide-up descriptions, tilt and Devanagari labels (निर्माण, सज्जा, शिल्प). Each links to `/services/[slug]`. |
| Process | Pinned horizontal timeline (Consultation → Design → Build → Furnish → Handover) with a progress rail; a vertical timeline on phones and for reduced motion |
| Portfolio | Filterable masonry (GSAP Flip), fullscreen lightbox (keyboard, swipe, focus trap) and a before/after drag slider |
| Furniture | Three.js chair viewer (rotate, pinch or Ctrl/⌘-scroll zoom, keyboard buttons, sheesham/teak/mango finish switcher), drag carousel, materials strip |
| Testimonials | Two opposing marquee rows that pause on hover or focus |
| Contact | Validated enquiry form (Indian mobile check, ₹ budget ranges), WhatsApp click-to-chat, Google Map embed, address |
| Footer | Jaali border, quick links, social icons, GSTIN/CIN placeholders |

Service detail pages (`/services/construction`, `/services/interiors`,
`/services/furniture`) add an included-scope grid, a gallery, FAQs (with
FAQPage schema) and a “next service” link.

## Motion and interaction

- Lenis smooth scrolling is driven by GSAP's ticker and synced with ScrollTrigger. Touch scrolling stays native.
- Terracotta curtain transitions for nav links between sections and pages (`TransitionLink`).
- Custom cursor that grows to show “View”, “Explore”, “Drag” or “Rotate” (`data-cursor="…"`), magnetic arch buttons and hover tilt. All three are fine-pointer only.
- Images fade from a blurred preview to sharp once loaded.
- **`prefers-reduced-motion`:** no smooth scroll, pinning, custom cursor or autoplay. Reveals are skipped and content is shown immediately; the slideshow stays manual.

## Accessibility and SEO

- Semantic landmarks, a skip link, a single `h1` per page, and alt text on every image (decorative images use `alt=""`).
- Keyboard support: the before/after slider is a `role="slider"`, the lightbox and mobile menu trap focus and close on Esc, the carousel is focusable and has buttons, and the 3D viewer has button controls.
- Text and focus-ring colours are chosen to meet WCAG AA contrast on their backgrounds; brass is used only for ornament.
- Metadata, Open Graph and Twitter cards, canonical URLs, `sitemap.xml`, `robots.txt`, a web manifest, and JSON-LD for `GeneralContractor` (LocalBusiness), `WebSite`, `Service`, `BreadcrumbList` and `FAQPage`.

## Project structure

```
app/                 routes, metadata files, API route (/api/enquiry)
components/
  layout/            header + mobile menu, footer, WhatsApp button
  motion/            preloader, cursor, magnetic, tilt, reveals, curtain link
  ornaments/         monogram, arch clip-paths, jaali, mandala
  sections/          page sections, lightbox, before/after, 3D viewer, carousel
  providers/         Lenis + curtain transition context
  ui/                SmartImage, ArchImage, ArchButton, icons
lib/
  site.ts            brand, contact and registration settings
  content.ts         all copy and data
  images.ts          placeholder image manifest
  three/             procedural chair model, textures and the viewer
assets/placeholders/ placeholder renders (replace)
```

## Deployment

Deploy on Vercel or any Node host (`npm run build && npm start`). The
enquiry endpoint is a Node route handler, so a fully static export is not
used; image optimisation also relies on the Next.js server.
