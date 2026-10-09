/**
 * Copy for the 3D house explorer. Kept free of three.js so the section
 * text renders immediately; the 3D model loads on demand.
 */

export const houseLayers = [
  { id: 'foundation', title: 'Foundation', hindi: 'नींव', text: 'A raft and column footings on compacted earth. The plinth lifts the house above the garden and keeps the monsoon out.' },
  { id: 'structure', title: 'Structure', hindi: 'ढाँचा', text: 'An RCC frame of columns, beams and slabs carries both floors, the stair and the 1.2 m cantilever.' },
  { id: 'walls', title: 'Walls', hindi: 'दीवारें', text: 'Coursed sandstone arcades on the ground floor, lime-plastered brick above, arched doorways and windows.' },
  { id: 'screens', title: 'Screens & glazing', hindi: 'जाली', text: 'Teak-framed glazing, timber fins, sandstone and brick jaalis, and a pergola that filter the sun.' },
  { id: 'interiors', title: 'Interiors', hindi: 'सज्जा', text: 'Kota stone and terrazzo floors, a terrazzo kitchen, block-printed rugs and brass pendants.' },
  { id: 'furniture', title: 'Furniture', hindi: 'शिल्प', text: 'Pieces from our workshop: cane chairs, a jhoola, charpai, chowki, a jaali bed and a sheesham dining set.' },
  { id: 'landscape', title: 'Landscape', hindi: 'बाग़', text: 'Pool, sandstone deck, lawn and shade trees complete the site and cool the breeze.' },
] as const;

export type HouseLayerId = (typeof houseLayers)[number]['id'];

/**
 * Design callouts. `layer` is the stage that builds them; `facing` is the
 * direction they face, so a callout hides when seen from behind.
 */
export const houseAnnotations = [
  { n: 1, title: 'Brick jaali screen', text: 'Shades the study from the west sun while letting the breeze through.', anchor: [0.25, 5.1, 1.4], facing: [0, 0.2, 1], layer: 3 },
  { n: 2, title: 'Sandstone arcade', text: 'A 2.6 m deep verandah keeps the ground floor in shade.', anchor: [-3.2, 2.0, 0.1], facing: [0, 0.2, 1], layer: 2 },
  { n: 3, title: 'Cantilevered upper floor', text: 'A 1.2 m overhang shades the glazing below.', anchor: [-7.7, 3.5, 1.2], facing: [-0.6, -0.1, 0.8], layer: 1 },
  { n: 4, title: 'Teak pergola terrace', text: 'Filtered shade for winter mornings above the dining room.', anchor: [5.0, 6.35, -2.2], facing: [0.3, 1, 0.5], layer: 3 },
  { n: 5, title: 'Pool court', text: 'Water cools the south breeze before it enters the house.', anchor: [-0.6, 0.1, 4.4], facing: [0, 1, 0.3], layer: 6 },
  { n: 6, title: 'West jaali windows', text: 'Star-shaped afternoon light moves across the living-room floor.', anchor: [-7.05, 1.7, -6.4], facing: [-1, 0.25, 0], layer: 3 },
] as const;

export const houseViews = [
  { id: 'eye', label: 'Eye level', text: 'True perspective from 1.6 m in the garden: the house as your eye will see it.' },
  { id: 'bird', label: "Bird's eye", text: 'The whole site from above, with the pool court and terrace.' },
  { id: 'axo', label: 'Axonometric', text: 'A parallel projection architects use to read volumes and depth without distortion.' },
  { id: 'plan', label: 'Plan', text: 'Looking straight down through a cut at 2 m, the way a floor plan is drawn.' },
  { id: 'elevation', label: 'Elevation', text: 'The south face, flat and to scale.' },
] as const;

export const houseSections = [
  { id: 'none', label: 'No cut' },
  { id: 'ground', label: 'Ground floor' },
  { id: 'upper', label: 'Upper floor' },
] as const;
