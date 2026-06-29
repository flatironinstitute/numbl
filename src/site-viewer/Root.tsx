import { useEffect, useState } from "react";
import { SiteApp } from "./SiteApp";
import { FigurePage } from "./FigurePage";

/** Re-render on hash changes so `#figure[/<id>]` can pick the route. */
function useHash(): string {
  const [hash, setHash] = useState(() => window.location.hash);
  useEffect(() => {
    const onHash = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  return hash;
}

/**
 * The static deploy has two views in one bundle:
 *   `/`            → the full IDE (SiteApp)
 *   `#figure`      → the first declared figure view
 *   `#figure/<id>` → a specific figure view by slug
 * Hash routing keeps deep links working on GitHub Pages without server config.
 */
export function Root() {
  const hash = useHash();
  const m = /^#figure(?:\/(.*))?$/.exec(hash);
  if (m) {
    const id = m[1] ? decodeURIComponent(m[1]) : undefined;
    return <FigurePage figureId={id} />;
  }
  return <SiteApp />;
}
