import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import { buildUihtmlSrcDoc } from "../graphics/uihtmlSrcDoc.js";

type UihtmlInstr = { type: "uihtml"; id: string; html: string; data?: string };

function uihtmlInstrs(code: string): UihtmlInstr[] {
  return executeCode(code).plotInstructions.filter(
    i => i.type === "uihtml"
  ) as UihtmlInstr[];
}

describe("uihtml Data bridge (runtime)", () => {
  it("uihtml('HTMLSource',h) with no Data emits no data field", () => {
    const instrs = uihtmlInstrs("uihtml('HTMLSource', '<p>hi</p>');");
    expect(instrs).toHaveLength(1);
    expect(instrs[0].html).toBe("<p>hi</p>");
    expect(instrs[0].data).toBeUndefined();
  });

  it("Data given at construction is jsonencode'd into the instruction", () => {
    const instrs = uihtmlInstrs(
      "uihtml('HTMLSource', '<p>hi</p>', 'Data', struct('a', 1, 'b', 2));"
    );
    expect(instrs).toHaveLength(1);
    expect(instrs[0].data).toBe('{"a":1,"b":2}');
  });

  it("string Data round-trips through jsonencode", () => {
    const instrs = uihtmlInstrs(
      "uihtml('HTMLSource', '<p>hi</p>', 'Data', 'Hello World!');"
    );
    expect(instrs[0].data).toBe('"Hello World!"');
  });

  it("h.Data = X; show(h) re-emits with the updated data", () => {
    const instrs = uihtmlInstrs(
      [
        "h = uihtml('HTMLSource', '<p>hi</p>', 'Data', struct('n', 1));",
        "h.Data = struct('n', 2);",
        "show(h);",
      ].join("\n")
    );
    expect(instrs).toHaveLength(2);
    expect(instrs[0].data).toBe('{"n":1}');
    expect(instrs[1].data).toBe('{"n":2}');
  });
});

describe("buildUihtmlSrcDoc", () => {
  it("injects the htmlComponent bridge bootstrap", () => {
    const doc = buildUihtmlSrcDoc("<p>hi</p>");
    expect(doc).toContain("<p>hi</p>");
    expect(doc).toContain("htmlComponent");
    expect(doc).toContain("DataChanged");
    expect(doc).toContain("window.setup");
    // No data → the payload literal is null.
    expect(doc).toContain("var payload = null;");
  });

  it("inserts the bootstrap before a closing </body>", () => {
    const doc = buildUihtmlSrcDoc("<html><body><p>hi</p></body></html>");
    expect(doc.indexOf("htmlComponent")).toBeLessThan(doc.indexOf("</body>"));
    expect(doc).toContain("</body></html>");
  });

  it("embeds the JSON data as a parseable payload", () => {
    const doc = buildUihtmlSrcDoc("<p>hi</p>", '{"a":1}');
    // The payload is the JSON text as a JS string literal that JSON.parse reads.
    expect(doc).toContain("JSON.parse(payload)");
    const m = doc.match(/var payload = (".*?");/);
    expect(m).not.toBeNull();
    // It must evaluate back to the original JSON text.
    expect(JSON.parse(m![1] as string)).toBe('{"a":1}');
  });

  it("neutralizes a </script> sequence hidden in the data", () => {
    const evil = JSON.stringify({ x: "</script><script>alert(1)</script>" });
    const doc = buildUihtmlSrcDoc("<p>hi</p>", evil);
    // The literal closing tag must not appear unescaped inside the payload.
    expect(doc).not.toContain("</script><script>alert(1)");
    expect(doc).toContain("\\u003c");
  });
});
