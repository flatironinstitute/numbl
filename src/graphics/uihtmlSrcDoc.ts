/**
 * Builds the iframe `srcdoc` for a `uihtml` component (MATLAB `uihtml`).
 *
 * The user's HTMLSource is left intact; a small bootstrap `<script>` is
 * injected that provides the standard MATLAB `htmlComponent` JavaScript object
 * and wires up the `Data` bridge so the same `.m` runs in numbl and real MATLAB.
 *
 * The numbl-vscode extension syncs src/graphics/ via devel/sync-graphics.sh.
 */

/** Encode `s` as a JavaScript string literal that is safe to embed inside an
 *  inline `<script>`: escapes `<`/`>`/`&` (so `</script>` and entities can't
 *  break out of the tag) and the line/paragraph separators that are illegal in
 *  JS string literals. */
function jsStringLiteral(s: string): string {
  return JSON.stringify(s)
    .replace(/</g, "\\u003c")
    .replace(/>/g, "\\u003e")
    .replace(/&/g, "\\u0026")
    .replace(/\u2028/g, "\\u2028")
    .replace(/\u2029/g, "\\u2029");
}

/** Build the iframe `srcdoc` for a `uihtml` component. The user's HTML is left
 *  intact; a small bootstrap `<script>` is appended that provides the standard
 *  MATLAB `htmlComponent` JavaScript object, calls the page's
 *  `setup(htmlComponent)` (if defined), and then — if `Data` was supplied —
 *  parses it and pushes it onto `htmlComponent.Data`, firing the page's
 *  `"DataChanged"` listeners. This mirrors MATLAB's `h.Data` → JavaScript path.
 *
 *  The reverse direction (JavaScript → MATLAB: `sendEventToMATLAB`, the `Data`
 *  setter's `DataChangedFcn`) has no live numbl callback once the script has
 *  finished, so those just post a message to the parent for any future host. */
export function buildUihtmlSrcDoc(html: string, data?: string): string {
  const dataLiteral = data === undefined ? "null" : jsStringLiteral(data);
  const bootstrap = `
<script>
(function () {
  var _data;
  var _listeners = Object.create(null);
  function _emit(name, ev) {
    var arr = _listeners[name];
    if (!arr) return;
    arr.slice().forEach(function (fn) {
      try { fn(ev); } catch (e) { console.error(e); }
    });
  }
  var htmlComponent = {
    addEventListener: function (name, fn) {
      (_listeners[name] || (_listeners[name] = [])).push(fn);
    },
    removeEventListener: function (name, fn) {
      var arr = _listeners[name];
      if (!arr) return;
      var i = arr.indexOf(fn);
      if (i >= 0) arr.splice(i, 1);
    },
    sendEventToMATLAB: function (name, data) {
      try {
        if (window.parent) window.parent.postMessage(
          { source: "numbl-uihtml", kind: "htmlEvent", name: name, data: data }, "*");
      } catch (e) {}
    }
  };
  Object.defineProperty(htmlComponent, "Data", {
    get: function () { return _data; },
    set: function (v) {
      _data = v;
      try {
        if (window.parent) window.parent.postMessage(
          { source: "numbl-uihtml", kind: "dataChanged", data: v }, "*");
      } catch (e) {}
    },
    enumerable: true,
    configurable: true
  });
  function _pushData(v) {
    var prev = _data;
    _data = v;
    _emit("DataChanged", {
      Data: v, PreviousData: prev, Source: htmlComponent, EventName: "DataChanged"
    });
  }
  function boot() {
    if (typeof window.setup === "function") {
      try { window.setup(htmlComponent); } catch (e) { console.error(e); }
    }
    var payload = ${dataLiteral};
    if (payload !== null) {
      var parsed;
      try { parsed = JSON.parse(payload); } catch (e) { parsed = payload; }
      _pushData(parsed);
    }
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
</script>`;
  // Insert before the final </body> if present (so it runs after the page's own
  // scripts); otherwise append.
  const idx = html.toLowerCase().lastIndexOf("</body>");
  if (idx === -1) return html + bootstrap;
  return html.slice(0, idx) + bootstrap + html.slice(idx);
}
