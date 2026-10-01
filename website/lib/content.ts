import type { StaticImageData } from 'next/image';
import { img } from './images';
import { site } from './site';

/* ──────────────────────────────────────────────────────────── navigation */

export const nav = [
  { label: 'About', href: '/#about' },
  { label: 'Services', href: '/#services' },
  { label: 'Process', href: '/#process' },
  { label: 'Work', href: '/#work' },
  { label: 'Furniture', href: '/#furniture' },
  { label: 'Contact', href: '/#contact' },
] as const;

/* ────────────────────────────────────────────────────────────────── hero */

export const hero = {
  eyebrow: `Design · Build · Furnish — ${site.city}, India`,
  lines: [['Built', 'to', 'last.'], ['Designed', 'to', 'belong.']],
  sub: 'Architecture, interiors and handcrafted furniture from one studio, with one team carrying your home from foundation to final furnishing.',
  slides: [
    { image: img.heroVilla, title: 'Pool Pavilion Villa', place: 'Jodhpur', alt: 'Sandstone villa with an arched verandah, brick jaali screen and pool at golden hour' },
    { image: img.heroLiving, title: 'Jaali Light Living Room', place: 'Jaipur', alt: 'Living room with arched jaali windows casting star-shaped light across a terrazzo floor' },
    { image: img.heroCourtyard, title: 'Aangan Courtyard House', place: 'Jaipur', alt: 'Open courtyard ringed by arched verandahs with terracotta walls and a neem tree' },
    { image: img.heroDining, title: 'Teak Jaali Dining Room', place: 'Ahmedabad', alt: 'Dining room with a sheesham table and cane chairs lit through a carved teak jaali screen' },
  ],
};

/* ───────────────────────────────────────────────────────────────── about */

export const about = {
  eyebrow: `Since ${site.foundingYear}`,
  eyebrowHi: 'हमारी कहानी',
  heading: 'Three crafts. One continuous hand.',
  paragraphs: [
    `We began as builders in the old lanes of ${site.city}, learning from masons who could read sandstone like a book. Over the years the work grew around the people who live in it: first the interiors, then the furniture we couldn't find anywhere else.`,
    'Today our architects, site engineers, interior designers and karigars work under one roof. Every home is conceived, built and furnished as a single idea, with one team accountable from the first sketch to the last cushion.',
  ],
  quote: 'A home should feel inevitable, as if the light, the stone and the people living there were always meant to be together.',
  stats: [
    { value: 350, suffix: '+', label: 'Projects completed' },
    { value: 12, suffix: ' Lakh+', label: 'Sq. ft. built' },
    { value: 600, suffix: '+', label: 'Happy families' },
    { value: 14, suffix: '', label: 'Cities across India' },
  ],
  images: {
    main: { src: img.aboutCourtyard, alt: 'View through an arched verandah into a sunlit courtyard with a neem tree' },
    detail: { src: img.aboutDetail, alt: 'Hand-woven cane lounge chair in a living room with jaali light' },
  },
};

/* ────────────────────────────────────────────────────────────── services */

export type ServiceSlug = 'construction' | 'interiors' | 'furniture';

export interface Service {
  slug: ServiceSlug;
  number: string;
  title: string;
  hindi: string;
  transliteration: string;
  tagline: string;
  summary: string;
  image: StaticImageData;
  imageAlt: string;
  intro: string;
  included: { title: string; text: string }[];
  highlights: { value: string; label: string }[];
  faqs: { q: string; a: string }[];
  category: Category;
}

export const services: Service[] = [
  {
    slug: 'construction',
    number: '01',
    title: 'Construction',
    hindi: 'निर्माण',
    transliteration: 'Nirmāṇ',
    tagline: 'Homes built on honest foundations.',
    summary:
      'Turnkey residences and boutique commercial buildings: structure, services and finishes, delivered by our own engineers and site teams.',
    image: img.serviceConstruction,
    imageAlt: 'Two-storey sandstone and lime-plaster villa with an arched verandah',
    intro:
      'We design and build homes the way they used to be built, slowly and carefully, with materials that age well. Our architects and engineers work alongside master masons, so the drawing and the building never drift apart.',
    included: [
      { title: 'Architecture & structure', text: 'Site analysis, Vastu-aware planning, structural design and detailed working drawings.' },
      { title: 'Approvals & liaison', text: 'Municipal sanctions, RERA documentation and utility connections handled for you.' },
      { title: 'RCC & masonry', text: 'Framed structures, load-bearing stone and brick, lime plaster and natural finishes.' },
      { title: 'Services & systems', text: 'Electrical, plumbing, solar, rainwater harvesting and smart-home wiring, all designed in.' },
      { title: 'Stone & craft', text: 'Dholpur, Jaisalmer and Kota stone, terracotta jaali and hand-finished details.' },
      { title: 'Transparent delivery', text: 'A fixed-scope BOQ, milestone billing and weekly photo reports from site.' },
    ],
    highlights: [
      { value: '5-year', label: 'structural warranty' },
      { value: 'Weekly', label: 'site reports' },
      { value: '1', label: 'point of contact' },
    ],
    faqs: [
      { q: 'How long does a new home take to build?', a: 'A typical 3,000–5,000 sq. ft. residence takes 12–16 months from groundbreaking, after 8–10 weeks of design.' },
      { q: 'Do you work outside your home city?', a: 'Yes. We currently build across 14 cities, with a resident site engineer on every project.' },
      { q: 'Can you work with our own architect?', a: 'Absolutely. We regularly build to other architects’ drawings, coordinating closely through construction.' },
      { q: 'How is pricing structured?', a: 'After design, you receive an itemised bill of quantities. Payments follow construction milestones, never ahead of them.' },
    ],
    category: 'Construction',
  },
  {
    slug: 'interiors',
    number: '02',
    title: 'Interior Design',
    hindi: 'सज्जा',
    transliteration: 'Sajjā',
    tagline: 'Rooms that hold light and life.',
    summary:
      'Complete interiors covering space planning, lighting, joinery, materials and styling, rooted in Indian craft and shaped around modern life.',
    image: img.serviceInteriors,
    imageAlt: 'Living room with terracotta niches, a linen sofa and a brass pendant',
    intro:
      'Good interiors are mostly about light, proportion and touch. We plan rooms around how you live, then layer in material, craft and colour until the space feels unmistakably yours.',
    included: [
      { title: 'Space planning', text: 'Layouts, furniture plans and circulation studied in detail before anything is built.' },
      { title: '3D visualisation', text: 'Photoreal views and material boards so every decision is made with confidence.' },
      { title: 'Lighting design', text: 'Layered daylight and artificial lighting, from jaali screens to concealed coves.' },
      { title: 'Joinery & kitchens', text: 'Wardrobes, kitchens and built-ins made in our own workshop.' },
      { title: 'Materials & finishes', text: 'Terrazzo, lime plaster, natural stone, handmade tiles and textiles.' },
      { title: 'Styling & art', text: 'Rugs, ceramics, artwork and plants chosen and placed for the final reveal.' },
    ],
    highlights: [
      { value: '10–16', label: 'weeks, typical apartment' },
      { value: 'In-house', label: 'joinery workshop' },
      { value: '3D', label: 'views before execution' },
    ],
    faqs: [
      { q: 'Do you take on single rooms?', a: 'We focus on whole homes and complete floors, where the design can work as one, but we are happy to discuss smaller projects.' },
      { q: 'Can we reuse our existing furniture?', a: 'Often, yes. We catalogue what you love and design around it, restoring pieces where needed.' },
      { q: 'How do you charge for design?', a: 'Design fees are fixed per project and adjusted against execution if we deliver the interiors.' },
      { q: 'Do you handle civil changes?', a: 'Yes. Our construction team handles all civil, electrical and plumbing work in-house.' },
    ],
    category: 'Interiors',
  },
  {
    slug: 'furniture',
    number: '03',
    title: 'Custom Furniture',
    hindi: 'शिल्प',
    transliteration: 'Śilp',
    tagline: 'Made by hand. Made for you.',
    summary:
      'Furniture designed in our studio and made in our workshop in solid sheesham, teak and mango wood, with hand-woven cane and brass.',
    image: img.serviceFurniture,
    imageAlt: 'Sheesham lounge chair with hand-woven cane and brass-capped legs',
    intro:
      'Our workshop is where the studio’s ideas become things you can touch. Every piece is drawn for its room, made by hand from solid timber and finished to be used, repaired and handed down.',
    included: [
      { title: 'Made to measure', text: 'Every piece sized and detailed for its room, and the people who will use it.' },
      { title: 'Solid wood joinery', text: 'Mortise-and-tenon and dovetail joints in seasoned sheesham, teak and mango wood.' },
      { title: 'Hand-woven cane', text: 'Traditional seven-step cane and niwar weaving by master weavers.' },
      { title: 'Brass detailing', text: 'Inlays, ferrules and hardware cast and finished by hand.' },
      { title: 'Natural upholstery', text: 'Linen, khadi and handloom cottons with natural latex and coir fills.' },
      { title: 'Restoration', text: 'Heirloom pieces repaired, re-caned and refinished with care.' },
    ],
    highlights: [
      { value: '6–8', label: 'weeks, made to order' },
      { value: '100%', label: 'solid timber' },
      { value: 'Lifetime', label: 'joinery guarantee' },
    ],
    faqs: [
      { q: 'Can we visit the workshop?', a: 'Please do. Visits are by appointment so a craftsman can walk you through the process.' },
      { q: 'Do you deliver outside your city?', a: 'Yes, we deliver and install across India with blanket-wrapped, crate-free packing.' },
      { q: 'Is the timber responsibly sourced?', a: 'We use certified plantation teak, reclaimed sheesham and mango wood from trees past fruiting age.' },
      { q: 'Can you match an existing piece?', a: 'Yes. Bring us a photograph or the piece itself and we will draw and sample it.' },
    ],
    category: 'Furniture',
  },
];

export const serviceBySlug = (slug: string) => services.find((s) => s.slug === slug);

/* ─────────────────────────────────────────────────────────────── process */

export const journey = {
  eyebrow: 'Our process',
  eyebrowHi: 'प्रक्रिया',
  heading: 'How a home comes together',
  intro: 'Five clear stages, one team and no hand-offs. You always know what happens next, and who is doing it.',
  steps: [
    { title: 'Consultation', hindi: 'परामर्श', time: 'Weeks 1–2', image: img.processConsult, alt: 'Massing model, drawings and material samples on a sheesham desk', text: 'We listen first. A site visit, a conversation about how you live, your budget and your timeline, and an honest view of what is possible.' },
    { title: 'Design', hindi: 'रूपांकन', time: 'Weeks 3–10', image: img.processDesign, alt: 'Architectural elevation drawing with arches and a brick jaali', text: 'Concept, plans, 3D views and a detailed bill of quantities. You will know how your home looks, and what it costs, before a single brick is laid.' },
    { title: 'Build', hindi: 'निर्माण', time: 'Months 3–14', image: img.processBuild, alt: 'Concrete frame under construction with bamboo scaffolding', text: 'Our own engineers and craftsmen on site, weekly progress reports and transparent milestones. No surprises.' },
    { title: 'Furnish', hindi: 'सज्जा', time: 'Final 8–12 weeks', image: img.processFurnish, alt: 'Dining room with cane chairs and pendant lights', text: 'Interiors installed and furniture made in our workshop, finished in place and styled down to the last cushion.' },
    { title: 'Handover', hindi: 'गृहप्रवेश', time: 'Move-in day', image: img.processHandover, alt: 'Carved teak door with a marigold toran, diyas and rangoli', text: 'A final walkthrough, a care guide for every material and a five-year structural warranty. The griha pravesh is yours.' },
  ],
};

/* ───────────────────────────────────────────────────────────── portfolio */

export type Category = 'Construction' | 'Interiors' | 'Furniture';
export const categories: ('All' | Category)[] = ['All', 'Construction', 'Interiors', 'Furniture'];

export interface Project {
  id: string;
  title: string;
  place: string;
  year: number;
  category: Category;
  image: StaticImageData;
  alt: string;
  shape?: 'arch' | 'rect';
}

// PLACEHOLDER projects: names, places and years are illustrative.
export const projects: Project[] = [
  { id: 'pool-pavilion', title: 'Pool Pavilion Villa', place: 'Jodhpur', year: 2025, category: 'Construction', image: img.heroVilla, alt: 'Sandstone villa with arched verandah and pool', shape: 'rect' },
  { id: 'living-jaali', title: 'Jaali Light Living Room', place: 'Jaipur', year: 2025, category: 'Interiors', image: img.heroLiving, alt: 'Living room with arched jaali windows and terrazzo floor', shape: 'rect' },
  { id: 'jharokha-chair', title: 'Jharokha Lounge Chair', place: 'Workshop', year: 2024, category: 'Furniture', image: img.serviceFurniture, alt: 'Sheesham and cane lounge chair', shape: 'arch' },
  { id: 'jaali-house', title: 'The Jaali House', place: 'Udaipur', year: 2024, category: 'Construction', image: img.workJaaliHouse, alt: 'Lime-plaster volume with a terracotta brick jaali screen', shape: 'arch' },
  { id: 'dining-teak', title: 'Teak Jaali Dining Room', place: 'Ahmedabad', year: 2024, category: 'Interiors', image: img.heroDining, alt: 'Dining room with carved teak jaali partition', shape: 'rect' },
  { id: 'charpai', title: 'Charpai Daybed', place: 'Workshop', year: 2024, category: 'Furniture', image: img.productCharpai, alt: 'Turned-leg sheesham charpai with niwar weave', shape: 'rect' },
  { id: 'courtyard', title: 'Aangan Courtyard House', place: 'Jaipur', year: 2023, category: 'Construction', image: img.heroCourtyard, alt: 'Courtyard house with arched verandahs and terracotta walls', shape: 'rect' },
  { id: 'arched-retreat', title: 'Arched Living Retreat', place: 'Pune', year: 2023, category: 'Interiors', image: img.aboutDetail, alt: 'Cane lounge chair beside a rug in a sunlit living room', shape: 'arch' },
  { id: 'jaali-console', title: 'Jaali Console in Teak', place: 'Workshop', year: 2023, category: 'Furniture', image: img.productConsole, alt: 'Teak console with hand-cut jaali doors', shape: 'rect' },
  { id: 'arcade', title: 'Sandstone Arcade Residence', place: 'Jaipur', year: 2023, category: 'Construction', image: img.workVillaArcade, alt: 'Coursed sandstone arcade with deep verandah', shape: 'rect' },
  { id: 'jhoola', title: 'Brass-Chain Jhoola', place: 'Workshop', year: 2022, category: 'Furniture', image: img.productJhoola, alt: 'Sheesham swing hung on brass chains', shape: 'arch' },
  { id: 'living-niche', title: 'Terracotta Niche Residence', place: 'Mumbai', year: 2022, category: 'Interiors', image: img.serviceInteriors, alt: 'Living room with terracotta-washed arched niches', shape: 'arch' },
  { id: 'site', title: 'Vaishali Nagar Residence: in progress', place: 'Jaipur', year: 2026, category: 'Construction', image: img.workSite, alt: 'Two-storey frame with bamboo scaffolding on site', shape: 'rect' },
  { id: 'sideboard', title: 'Reeded Mango-Wood Sideboard', place: 'Workshop', year: 2022, category: 'Furniture', image: img.productSideboard, alt: 'Mango wood sideboard with reeded doors and brass pulls', shape: 'rect' },
];

export const beforeAfter = {
  title: 'Malviya Nagar Residence',
  place: 'Jaipur',
  text: 'A bare shell turned into a light-filled living room with jaali windows, terrazzo floors, terracotta niches and furniture from our workshop.',
  before: { src: img.beforeLiving, alt: 'Before: unfinished room with raw cement walls and a bare floor' },
  after: { src: img.afterLiving, alt: 'After: finished living room with jaali windows, terrazzo floor and handcrafted furniture' },
};

/* ───────────────────────────────────────────────────────────── furniture */

export const featuredPiece = {
  name: 'Jharokha Lounge Chair',
  hindi: 'झरोखा',
  text: 'Named for the overhanging windows of Rajasthani havelis. A turned-leg frame, an arched back of hand-woven cane and brass ferrules that age with you.',
  specs: [
    { label: 'Dimensions', value: 'W 72 × D 78 × H 92 cm' },
    { label: 'Seat height', value: '42 cm' },
    { label: 'Lead time', value: 'Made to order · 6–8 weeks' },
  ],
};

export const products = [
  { name: 'Haveli Dining Chair', materials: 'Sheesham · cane seat · carved jaali back', image: img.productDiningChair },
  { name: 'Charpai Daybed', materials: 'Sheesham · turned legs · hand-woven niwar', image: img.productCharpai },
  { name: 'Baithak Chowki', materials: 'Mango wood · brass inlay', image: img.productChowki },
  { name: 'Jaali Console', materials: 'Teak · hand-cut jaali doors · brass', image: img.productConsole },
  { name: 'Reeded Sideboard', materials: 'Mango wood · reeded fronts · brass pulls', image: img.productSideboard },
  { name: 'Jhoola Swing', materials: 'Sheesham · brass chains · cotton cushions', image: img.productJhoola },
  { name: 'Mudha Stool', materials: 'Sarkanda reed · jute · cotton', image: img.productMudha },
];

export const materials = [
  { name: 'Sheesham', note: 'Indian rosewood', text: 'Dense, dark and richly figured. Our choice for frames that will outlive us.', image: img.materialSheesham },
  { name: 'Teak', note: 'Plantation grown', text: 'Golden, oily and weather-proof. The timber of havelis and ships alike.', image: img.materialTeak },
  { name: 'Mango wood', note: 'Reclaimed hardwood', text: 'From trees past their fruiting years, with a lively, pale grain.', image: img.materialMango },
  { name: 'Cane', note: 'Hand-woven rattan', text: 'The classic seven-step weave: breathable, light and endlessly repairable.', image: img.materialCane },
  { name: 'Brass', note: 'Hand-finished', text: 'Inlays, ferrules and hardware that age into a warm patina.', image: img.materialBrass },
];

/* ────────────────────────────────────────────────────────── testimonials */

// PLACEHOLDER testimonials: replace with real client quotes, used with permission.
export const testimonials = [
  { quote: 'They built our home and then made the dining table we eat at every night. Rare to find a team that cares about the foundation and the finish equally.', name: 'Ananya & Rohit M.', city: 'Jaipur' },
  { quote: 'Weekly site reports, honest costing and not a single surprise in fourteen months. We would do it all again.', name: 'Vikram S.', city: 'Jodhpur' },
  { quote: 'The jaali wall throws a new pattern across the floor every hour. Guests still ask who designed it.', name: 'Priya N.', city: 'Bengaluru' },
  { quote: 'Our forty-year-old flat feels brand new, and somehow more like us than before.', name: 'Farah & Imran Q.', city: 'Mumbai' },
  { quote: 'The cane chairs are lighter than they look and sturdier than anything we have owned.', name: 'Meera I.', city: 'Chennai' },
  { quote: 'From the first sketch to griha pravesh, one team and one point of contact. That alone was worth it.', name: 'Sanjay & Kavita A.', city: 'New Delhi' },
  { quote: 'They restored my grandmother’s haveli doors and set them into a completely modern home.', name: 'Aditya S.', city: 'Udaipur' },
  { quote: 'Thoughtful, unhurried and precise. One visit to the workshop and we were sold.', name: 'Nandini R.', city: 'Pune' },
];

/* ─────────────────────────────────────────────────────────────── contact */

export const enquiry = {
  serviceOptions: ['Construction', 'Interior Design', 'Custom Furniture', 'End-to-end: design, build & furnish'],
  budgetOptions: ['Under ₹25 Lakh', '₹25 – 50 Lakh', '₹50 Lakh – ₹1 Crore', '₹1 – 3 Crore', 'Above ₹3 Crore', 'Not sure yet'],
};
