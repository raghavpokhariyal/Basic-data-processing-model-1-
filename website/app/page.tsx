import { Hero } from '@/components/sections/Hero';
import { About } from '@/components/sections/About';
import { Services } from '@/components/sections/Services';
import { Process } from '@/components/sections/Process';
import { HouseExplorer } from '@/components/sections/HouseExplorer';
import { Portfolio } from '@/components/sections/Portfolio';
import { Furniture } from '@/components/sections/Furniture';
import { Testimonials } from '@/components/sections/Testimonials';
import { Contact } from '@/components/sections/Contact';

export default function HomePage() {
  return (
    <>
      <Hero />
      <About />
      <Services />
      <Process />
      <HouseExplorer />
      <Portfolio />
      <Furniture />
      <Testimonials />
      <Contact />
    </>
  );
}
