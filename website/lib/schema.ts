import { services } from './content';
import { site } from './site';

/** LocalBusiness (GeneralContractor) schema for the studio. */
export function localBusinessSchema() {
  return {
    '@context': 'https://schema.org',
    '@type': ['GeneralContractor', 'HomeAndConstructionBusiness'],
    '@id': `${site.url}/#business`,
    name: site.name,
    legalName: site.legalName,
    description: site.description,
    url: site.url,
    logo: `${site.url}/icon.svg`,
    image: `${site.url}/opengraph-image.jpg`,
    telephone: site.phone.e164,
    email: site.email,
    priceRange: '₹₹₹',
    foundingDate: String(site.foundingYear),
    founder: { '@type': 'Person', name: site.founder.name, jobTitle: site.founder.role },
    address: {
      '@type': 'PostalAddress',
      streetAddress: site.address.street,
      addressLocality: site.address.locality,
      addressRegion: site.address.region,
      postalCode: site.address.postalCode,
      addressCountry: site.address.countryCode,
    },
    geo: { '@type': 'GeoCoordinates', latitude: site.geo.latitude, longitude: site.geo.longitude },
    areaServed: [{ '@type': 'City', name: site.city }, { '@type': 'Country', name: 'India' }],
    openingHours: site.hours.schema,
    sameAs: site.social.map((s) => s.href),
    hasOfferCatalog: {
      '@type': 'OfferCatalog',
      name: 'Design & build services',
      itemListElement: services.map((s) => ({
        '@type': 'Offer',
        itemOffered: { '@type': 'Service', name: s.title, description: s.summary, url: `${site.url}/services/${s.slug}` },
      })),
    },
  };
}

export function websiteSchema() {
  return {
    '@context': 'https://schema.org',
    '@type': 'WebSite',
    '@id': `${site.url}/#website`,
    name: site.name,
    url: site.url,
    inLanguage: 'en-IN',
    publisher: { '@id': `${site.url}/#business` },
  };
}
