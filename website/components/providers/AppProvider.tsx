'use client';

import Lenis from 'lenis';
import { usePathname, useRouter } from 'next/navigation';
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { gsap, ScrollTrigger, prefersReducedMotion } from '@/lib/gsap';
import { Monogram } from '@/components/ornaments/Monogram';

interface AppContextValue {
  lenis: Lenis | null;
  /** True once the page is visible (preloader finished / curtain lifting). */
  revealed: boolean;
  markRevealed: () => void;
  /** Navigate with the terracotta curtain wipe (same-page hashes or routes). */
  navigate: (href: string) => void;
  lockScroll: (locked: boolean) => void;
}

const AppContext = createContext<AppContextValue | null>(null);

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used inside <AppProvider>');
  return ctx;
}

function scrollTarget(hash: string) {
  if (!hash || hash === '#') return 0;
  try {
    return document.querySelector<HTMLElement>(hash) ?? 0;
  } catch {
    return 0;
  }
}

export function AppProvider({ children }: { children: ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [lenis, setLenis] = useState<Lenis | null>(null);
  const [revealed, setRevealed] = useState(false);
  const curtainRef = useRef<HTMLDivElement>(null);
  const pendingHash = useRef<string | null>(null);
  const busy = useRef(false);
  const lenisRef = useRef<Lenis | null>(null);

  // Smooth scrolling (skipped entirely for reduced motion; touch stays native)
  useEffect(() => {
    if (prefersReducedMotion()) return;
    const instance = new Lenis({
      duration: 1.15,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smoothWheel: true,
      autoRaf: false,
    });
    instance.on('scroll', ScrollTrigger.update);
    const tick = (time: number) => instance.raf(time * 1000);
    gsap.ticker.add(tick);
    gsap.ticker.lagSmoothing(0);
    lenisRef.current = instance;
    setLenis(instance);
    return () => {
      gsap.ticker.remove(tick);
      instance.destroy();
      lenisRef.current = null;
    };
  }, []);

  const jumpTo = useCallback((hash: string) => {
    const target = scrollTarget(hash);
    const l = lenisRef.current;
    if (l) l.scrollTo(target, { immediate: true, force: true });
    else if (target === 0) window.scrollTo(0, 0);
    else target.scrollIntoView({ behavior: 'auto', block: 'start' });
  }, []);

  const lockScroll = useCallback((locked: boolean) => {
    document.documentElement.classList.toggle('is-locked', locked);
    const l = lenisRef.current;
    if (l) {
      if (locked) l.stop();
      else l.start();
    }
  }, []);

  const lift = useCallback(() => {
    const el = curtainRef.current;
    setRevealed(true);
    if (!el) return;
    gsap.to(el, {
      yPercent: -100,
      duration: 0.95,
      ease: 'expo.inOut',
      delay: 0.05,
      onComplete: () => {
        gsap.set(el, { display: 'none' });
        busy.current = false;
      },
    });
  }, []);

  // Compare against the router's pathname rather than window.location, so
  // navigation also works where the page URL itself never changes.
  const pathRef = useRef(pathname);
  pathRef.current = pathname;

  const navigate = useCallback(
    (href: string) => {
      const [rawPath, rawHash] = href.split('#');
      const url = { pathname: rawPath || pathRef.current, hash: rawHash ? `#${rawHash}` : '' };
      const samePage = url.pathname === pathRef.current;
      if (samePage && url.hash) {
        try {
          history.replaceState(null, '', url.hash);
        } catch {
          /* history unavailable (sandboxed frame): the scroll still happens */
        }
      }

      if (prefersReducedMotion() || !curtainRef.current) {
        if (samePage) {
          const t = scrollTarget(url.hash);
          if (t === 0) window.scrollTo(0, 0);
          else t.scrollIntoView({ block: 'start' });
        } else router.push(url.pathname + url.hash);
        return;
      }
      if (busy.current) return;
      busy.current = true;
      const el = curtainRef.current;
      gsap.killTweensOf(el);
      gsap.set(el, { display: 'block', yPercent: 120 });
      gsap.to(el, {
        yPercent: 0,
        duration: 0.8,
        ease: 'expo.inOut',
        onComplete: () => {
          if (samePage) {
            jumpTo(url.hash);
            ScrollTrigger.refresh();
            lift();
          } else {
            setRevealed(false);
            pendingHash.current = url.hash || '#';
            router.push(url.pathname + url.hash, { scroll: false });
          }
        },
      });
    },
    [jumpTo, lift, router],
  );

  // After a route change under the curtain: position, refresh, then lift
  useEffect(() => {
    if (pendingHash.current === null) return;
    const hash = pendingHash.current;
    pendingHash.current = null;
    requestAnimationFrame(() => {
      jumpTo(hash);
      ScrollTrigger.refresh();
      lift();
    });
  }, [pathname, jumpTo, lift]);

  const markRevealed = useCallback(() => setRevealed(true), []);

  const value = useMemo(
    () => ({ lenis, revealed, markRevealed, navigate, lockScroll }),
    [lenis, revealed, markRevealed, navigate, lockScroll],
  );

  return (
    <AppContext.Provider value={value}>
      {children}
      <div
        ref={curtainRef}
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 z-[96]"
        style={{ display: 'none', willChange: 'transform' }}
      >
        <div
          className="absolute inset-x-0 -top-[16vh] h-[16.2vh] bg-terracotta"
          style={{ borderRadius: '50% 50% 0 0 / 100% 100% 0 0' }}
        />
        <div className="absolute inset-0 bg-terracotta" />
        <div className="absolute inset-0 grid place-items-center">
          <Monogram className="h-24 w-auto text-ivory/85" />
        </div>
      </div>
    </AppContext.Provider>
  );
}
