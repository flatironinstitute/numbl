/** Host side of the uihtml iframe bridge (the page side is the bootstrap in
 *  uihtmlSrcDoc.ts). Each host forwards page events to its interpreter and
 *  relays `sendEventToHTMLSource` back into the matching iframe. */

/** An event a uihtml page posted (sendEventToMATLAB or a Data set). */
export interface UihtmlPageEvent {
  compId: string;
  kind: "htmlEvent" | "dataChanged";
  name?: string;
  data: unknown;
}

/** Listen for events from uihtml iframes in this window. Returns an
 *  unsubscribe function. */
export function listenForUihtmlEvents(
  onEvent: (ev: UihtmlPageEvent) => void
): () => void {
  const onMessage = (e: MessageEvent) => {
    const d = e.data;
    if (!d || d.source !== "numbl-uihtml") return;
    onEvent({ compId: d.compId, kind: d.kind, name: d.name, data: d.data });
  };
  window.addEventListener("message", onMessage);
  return () => window.removeEventListener("message", onMessage);
}

/** Deliver a script -> page event to every iframe rendering `compId`; the
 *  bootstrap fires the page's addEventListener(name) listeners. */
export function postToUihtmlFrames(
  compId: string,
  name: string,
  dataJson: string
): void {
  const frames = document.querySelectorAll<HTMLIFrameElement>(
    `iframe[title="uihtml-${compId}"]`
  );
  frames.forEach(f =>
    f.contentWindow?.postMessage(
      { source: "numbl-host", compId, name, dataJson },
      "*"
    )
  );
}
