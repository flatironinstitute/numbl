import { useEffect } from "react";

const SITE = "https://numbl.org";

function setMeta(selector: string, attr: string, value: string) {
  let el = document.head.querySelector<HTMLMetaElement | HTMLLinkElement>(
    selector
  );
  if (!el) {
    const isLink = selector.startsWith("link");
    el = document.createElement(isLink ? "link" : "meta");
    if (selector.includes('property="')) {
      el.setAttribute("property", selector.match(/property="([^"]+)"/)![1]);
    } else if (selector.includes('name="')) {
      el.setAttribute("name", selector.match(/name="([^"]+)"/)![1]);
    } else if (selector.includes('rel="')) {
      el.setAttribute("rel", selector.match(/rel="([^"]+)"/)![1]);
    }
    document.head.appendChild(el);
  }
  el.setAttribute(attr, value);
}

export interface PageMeta {
  title: string;
  description: string;
  /** Path portion of the canonical URL, e.g. "/docs/language". */
  path: string;
}

/**
 * Keep the document <head> in sync with the active route so client-side
 * navigation (and JS-rendering crawlers) see per-page title, description,
 * canonical, and Open Graph / Twitter tags. Build-time prerendering
 * (scripts/prerender-seo.ts) provides the same tags for the initial HTML.
 */
export function usePageMeta({ title, description, path }: PageMeta) {
  useEffect(() => {
    const url = path === "/" ? `${SITE}/` : `${SITE}${path}`;
    document.title = title;
    setMeta('meta[name="description"]', "content", description);
    setMeta('link[rel="canonical"]', "href", url);
    setMeta('meta[property="og:url"]', "content", url);
    setMeta('meta[property="og:title"]', "content", title);
    setMeta('meta[property="og:description"]', "content", description);
    setMeta('meta[name="twitter:title"]', "content", title);
    setMeta('meta[name="twitter:description"]', "content", description);
  }, [title, description, path]);
}
