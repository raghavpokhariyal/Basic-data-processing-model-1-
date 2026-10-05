/**
 * ─────────────────────────────────────────────────────────────────────────
 *  BRAND SETTINGS — edit this file first.
 *
 *  The brief left the brand name and city as [BRAND NAME] and [CITY].
 *  "Aangan" (आँगन, the Indian courtyard) and Jaipur are stand-in values so
 *  the design can be reviewed; replace them with the real details below.
 *  Every address, number and registration ID here is a placeholder.
 * ─────────────────────────────────────────────────────────────────────────
 */
export const site = {
  /** [BRAND NAME] */
  name: 'Aangan',
  /** Shown in the logo lock-up and preloader monogram */
  monogram: 'A',
  /** Decorative Devanagari rendering of the name */
  nameDevanagari: 'आँगन',
  legalName: 'Aangan Design & Build Pvt. Ltd.',
  tagline: 'Design · Build · Furnish',
  description:
    'A Jaipur design-and-build studio for construction, interior design and handcrafted furniture. One team, from foundation to final furnishing.',

  /** [CITY] */
  city: 'Jaipur',
  region: 'Rajasthan',
  country: 'India',
  address: {
    street: '[Studio address line], C-Scheme',
    locality: 'Jaipur',
    region: 'Rajasthan',
    postalCode: '302001',
    countryCode: 'IN',
  },
  /** Used for the map pin and LocalBusiness schema */
  geo: { latitude: 26.9124, longitude: 75.7873 },
  mapQuery: 'C-Scheme, Jaipur, Rajasthan 302001',

  url: process.env.NEXT_PUBLIC_SITE_URL ?? 'https://www.example.com',
  /** Display format and the digits-only form used for tel: and WhatsApp links */
  phone: { display: '+91 00000 00000', e164: '+910000000000' },
  whatsapp: { number: '910000000000', message: "Hello! I'd like to discuss a project with your studio." },
  email: 'hello@example.com',
  hours: { display: 'Mon – Sat · 10:00 – 19:00', schema: 'Mo-Sa 10:00-19:00' },

  foundingYear: 2008,
  founder: { name: '[Founder Name]', role: 'Founder & Principal Architect' },

  registration: {
    gstin: '08XXXXXXXXXXXZX',
    cin: 'U00000RJ2008PTC000000',
  },

  social: [
    { label: 'Instagram', href: 'https://www.instagram.com/your-handle' },
    { label: 'Pinterest', href: 'https://www.pinterest.com/your-handle' },
    { label: 'LinkedIn', href: 'https://www.linkedin.com/company/your-handle' },
    { label: 'YouTube', href: 'https://www.youtube.com/@your-handle' },
    { label: 'Facebook', href: 'https://www.facebook.com/your-handle' },
  ],
} as const;

export const whatsappHref = (message: string = site.whatsapp.message) =>
  `https://wa.me/${site.whatsapp.number}?text=${encodeURIComponent(message)}`;

export const mapEmbedSrc = `https://www.google.com/maps?q=${encodeURIComponent(site.mapQuery)}&z=15&output=embed`;
export const mapLinkHref = `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(site.mapQuery)}`;

export const yearsOfPractice = () => new Date().getFullYear() - site.foundingYear;
