import type { Metadata, Viewport } from 'next';
import { Cormorant_Garamond, Manrope, Tiro_Devanagari_Hindi } from 'next/font/google';
import 'lenis/dist/lenis.css';
import './globals.css';
import { AppProvider } from '@/components/providers/AppProvider';
import { Preloader } from '@/components/motion/Preloader';
import { Cursor } from '@/components/motion/Cursor';
import { Header } from '@/components/layout/Header';
import { Footer } from '@/components/layout/Footer';
import { WhatsAppFab } from '@/components/layout/WhatsAppFab';
import { ArchDefs } from '@/components/ornaments/ArchDefs';
import { JsonLd } from '@/components/seo/JsonLd';
import { localBusinessSchema, websiteSchema } from '@/lib/schema';
import { site } from '@/lib/site';

const serif = Cormorant_Garamond({
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  style: ['normal', 'italic'],
  variable: '--font-cormorant',
  display: 'swap',
});
const sans = Manrope({ subsets: ['latin'], variable: '--font-manrope', display: 'swap' });
const deva = Tiro_Devanagari_Hindi({ subsets: ['devanagari'], weight: '400', variable: '--font-tiro', display: 'swap' });

const title = `${site.name} | Architecture, Interiors & Custom Furniture in ${site.city}`;

export const metadata: Metadata = {
  metadataBase: new URL(site.url),
  title: { default: title, template: `%s | ${site.name}, ${site.city}` },
  description: site.description,
  applicationName: site.name,
  keywords: [
    `construction company ${site.city}`,
    `interior designer ${site.city}`,
    `custom furniture ${site.city}`,
    'design and build India',
    'turnkey home construction',
    'luxury interiors India',
    'sheesham furniture',
    'cane furniture',
    'jaali design',
  ],
  authors: [{ name: site.legalName }],
  alternates: { canonical: '/' },
  openGraph: {
    type: 'website',
    locale: 'en_IN',
    url: '/',
    siteName: site.name,
    title,
    description: site.description,
  },
  twitter: { card: 'summary_large_image', title, description: site.description },
  robots: { index: true, follow: true },
  formatDetection: { telephone: false },
  other: { 'geo.region': 'IN-RJ', 'geo.placename': site.city },
};

export const viewport: Viewport = {
  themeColor: '#f5f0e8',
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en-IN" className={`${serif.variable} ${sans.variable} ${deva.variable}`} suppressHydrationWarning>
      <body>
        <noscript>
          <style>{`.preloader{display:none!important}.smart-img{opacity:1!important;filter:none!important;transform:none!important}`}</style>
        </noscript>
        <JsonLd data={[localBusinessSchema(), websiteSchema()]} />
        <ArchDefs />
        <a
          href="#main"
          className="fixed top-3 left-3 z-[130] -translate-y-24 bg-indigo px-4 py-3 text-sm font-semibold text-ivory transition-transform focus:translate-y-0"
        >
          Skip to content
        </a>
        <AppProvider>
          <Preloader />
          <Header />
          <main id="main" tabIndex={-1} className="outline-none">
            {children}
          </main>
          <Footer />
          <WhatsAppFab />
          <Cursor />
        </AppProvider>
        <div className="grain" aria-hidden="true" />
      </body>
    </html>
  );
}
