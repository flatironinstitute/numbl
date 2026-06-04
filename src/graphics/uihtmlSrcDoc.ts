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
 *  The reverse direction (JavaScript → MATLAB) is also wired: `sendEventToMATLAB`
 *  and the `Data` setter post a message (tagged with `compId`) to the parent;
 *  the host relays it into the worker, which re-enters the interpreter to fire
 *  the registered `HTMLEventReceivedFcn` / `DataChangedFcn`. The host's
 *  `sendEventToHTMLSource` comes back as a `numbl-host` message that this
 *  bootstrap delivers to the page's `addEventListener(name)` listeners. */
export function buildUihtmlSrcDoc(
  html: string,
  data?: string,
  compId?: string
): string {
  const dataLiteral = data === undefined ? "null" : jsStringLiteral(data);
  const compIdLiteral = compId === undefined ? "null" : jsStringLiteral(compId);
  const bootstrap = `
<script>
(function () {
  var COMP_ID = ${compIdLiteral};
  var _data;
  var _listeners = Object.create(null);
  function _emit(name, ev) {
    var arr = _listeners[name];
    if (!arr) return;
    arr.slice().forEach(function (fn) {
      try { fn(ev); } catch (e) { console.error(e); }
    });
  }
  function _post(msg) {
    try {
      msg.source = "numbl-uihtml";
      msg.compId = COMP_ID;
      if (window.parent) window.parent.postMessage(msg, "*");
    } catch (e) {}
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
      _post({ kind: "htmlEvent", name: name, data: data });
    }
  };
  Object.defineProperty(htmlComponent, "Data", {
    get: function () { return _data; },
    set: function (v) {
      _data = v;
      // JS -> MATLAB: fires DataChangedFcn (does NOT fire local DataChanged).
      _post({ kind: "dataChanged", data: v });
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
  // MATLAB -> JS events (sendEventToHTMLSource) arrive as "numbl-host" messages.
  window.addEventListener("message", function (e) {
    var d = e.data;
    if (!d || d.source !== "numbl-host" || d.compId !== COMP_ID) return;
    var parsed;
    try { parsed = JSON.parse(d.dataJson); } catch (err) { parsed = d.dataJson; }
    _emit(d.name, {
      Data: parsed, Source: htmlComponent, EventName: d.name
    });
  });
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
