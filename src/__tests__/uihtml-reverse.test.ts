import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";

describe("uihtml reverse channel (HTML -> MATLAB)", () => {
  it("dispatches an HTML event into a callback and sends a result back", () => {
    const out: string[] = [];
    const outgoing: { compId: string; name: string; dataJson: string }[] = [];
    const code = [
      "uihtml('HTMLSource','<p>x</p>','HTMLEventReceivedFcn',@onEvent);",
      "function onEvent(src, ev)",
      "  disp(['handled:' ev.HTMLEventName]);",
      "  sendEventToHTMLSource(src, 'Result', ev.HTMLEventData ^ 2);",
      "end",
    ].join("\n");

    const result = executeCode(code, {
      onOutput: t => out.push(t),
      onHtmlSourceEvent: (compId, name, dataJson) =>
        outgoing.push({ compId, name, dataJson }),
    });

    // The run armed a session because a callback was registered.
    expect(result.uihtmlSession).toBeDefined();
    expect(result.uihtmlSession!.hasCallbacks()).toBe(true);

    // Simulate the page firing sendEventToMATLAB("Square", 7).
    result.uihtmlSession!.dispatchEvent("uh1", "HTMLEventReceived", {
      name: "Square",
      data: 7,
    });

    // The MATLAB callback ran (output) and sent a result back to the page.
    expect(out.join("")).toContain("handled:Square");
    expect(outgoing).toEqual([
      { compId: "uh1", name: "Result", dataJson: "49" },
    ]);

    // Disarm releases the callbacks.
    result.uihtmlSession!.dispose();
    expect(result.uihtmlSession!.hasCallbacks()).toBe(false);
  });

  it("no callbacks -> no session (zero overhead for normal uihtml)", () => {
    const result = executeCode("uihtml('HTMLSource','<p>hi</p>');");
    expect(result.uihtmlSession).toBeUndefined();
  });

  it("anonymous-function callback with a captured value, across events", () => {
    const out: string[] = [];
    const code = [
      "gain = 10;",
      "uihtml('HTMLSource','<p>x</p>','HTMLEventReceivedFcn', " +
        "@(src,ev) sendEventToHTMLSource(src,'Result', ev.HTMLEventData * gain));",
    ].join("\n");
    const r = executeCode(code, {
      onHtmlSourceEvent: (_c, _n, dataJson) => out.push(dataJson),
    });
    expect(r.uihtmlSession).toBeDefined();
    // The captured `gain` survives the run and every dispatch.
    r.uihtmlSession!.dispatchEvent("uh1", "HTMLEventReceived", {
      name: "S",
      data: 7,
    });
    r.uihtmlSession!.dispatchEvent("uh1", "HTMLEventReceived", {
      name: "S",
      data: 3,
    });
    expect(out).toEqual(["70", "30"]);
  });

  it("anonymous fn capturing a handle object accumulates state across events", () => {
    const out: string[] = [];
    const code = [
      "s = State();",
      "uihtml('HTMLSource','<p>x</p>','HTMLEventReceivedFcn', " +
        "@(src,ev) bump(src,ev,s));",
      "function bump(~, ev, s)",
      "  s.n = s.n + ev.HTMLEventData;",
      "  disp(s.n);",
      "end",
    ].join("\n");
    const r = executeCode(code, { onOutput: t => out.push(t) }, [
      {
        name: "State.m",
        source: "classdef State < handle\n properties\n  n = 0\n end\nend\n",
      },
    ]);
    // Mutation through the captured handle persists between callbacks: 2, then 7.
    r.uihtmlSession!.dispatchEvent("uh1", "HTMLEventReceived", {
      name: "S",
      data: 2,
    });
    r.uihtmlSession!.dispatchEvent("uh1", "HTMLEventReceived", {
      name: "S",
      data: 5,
    });
    expect(out.join("")).toBe("2\n7\n");
  });

  it("DataChanged callback fires with the parsed data", () => {
    const out: string[] = [];
    const code = [
      "uihtml('HTMLSource','<p>x</p>','DataChangedFcn',@onData);",
      "function onData(src, ev)",
      "  disp(ev.Data.label);",
      "end",
    ].join("\n");
    const result = executeCode(code, { onOutput: t => out.push(t) });
    expect(result.uihtmlSession).toBeDefined();
    result.uihtmlSession!.dispatchEvent("uh1", "DataChanged", {
      data: { label: "hi" },
    });
    expect(out.join("")).toContain("hi");
  });
});
