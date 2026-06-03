import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import {
  figuresReducer,
  initialFiguresState,
  type FiguresState,
} from "../graphics/figuresReducer.js";
import type { AxesState } from "../graphics/figuresReducer.js";

/** Reduce all emitted plot instructions (as the viewer does) and return the
 *  current axes — the rendered state after the script runs. Without `--plot`,
 *  drawnow is a no-op so every instruction accumulates in plotInstructions. */
function renderedAxes(code: string): AxesState {
  const result = executeCode(code);
  let state: FiguresState = initialFiguresState;
  for (const instr of result.plotInstructions) {
    state = figuresReducer(state, instr);
  }
  const fig = state.figs[state.currentHandle];
  return fig.axes[fig.currentAxesIndex];
}

describe("set on graphics handles", () => {
  it("set(h,'XData',v,'YData',v) updates the rendered line trace", () => {
    const axes = renderedAxes(
      "f = line([0 1],[0 0]); set(f,'xdata',[0 5],'ydata',[2 3]);"
    );
    expect(axes.traces).toHaveLength(1);
    expect(axes.traces[0].x).toEqual([0, 5]);
    expect(axes.traces[0].y).toEqual([2, 3]);
  });

  it("scalar XData/YData become single-point arrays", () => {
    const axes = renderedAxes(
      "f = line(0,0,'marker','.'); set(f,'xdata',3,'ydata',4);"
    );
    expect(axes.traces[0].x).toEqual([3]);
    expect(axes.traces[0].y).toEqual([4]);
  });

  it("set updates color/linewidth on the rendered trace", () => {
    const axes = renderedAxes(
      "f = line([0 1],[0 1]); set(f,'Color','r','LineWidth',5);"
    );
    expect(axes.traces[0].color).toEqual([1, 0, 0]);
    expect(axes.traces[0].lineWidth).toBe(5);
  });

  it("emits exactly one update_trace per set with changes", () => {
    const result = executeCode(
      "f = line([0 1],[0 0]); set(f,'xdata',[1 2],'ydata',[3 4]);"
    );
    const updates = result.plotInstructions.filter(
      i => i.type === "update_trace"
    );
    expect(updates).toHaveLength(1);
    expect((updates[0] as { id: number }).id).toBe(1);
    expect((updates[0] as { props: Record<string, unknown> }).props).toEqual({
      x: [1, 2],
      y: [3, 4],
    });
  });

  it("set(gca,...) and set on non-handles are accepted no-ops", () => {
    const result = executeCode("set(gca,'FontSize',14); set(0,'Color','r');");
    expect(
      result.plotInstructions.filter(i => i.type === "update_trace")
    ).toHaveLength(0);
  });

  it("tt = title(...); set(tt,'String',...) updates the rendered title", () => {
    const axes = renderedAxes(
      "tt = title('start'); set(tt,'String','updated');"
    );
    expect(axes.title).toBe("updated");
  });

  it("title handle exposes String for reading", () => {
    const result = executeCode("tt = title('hello'); s = tt.String;");
    const s = result.variableValues["s"] as { value: string };
    expect(s.value).toBe("hello");
  });

  it("animation loop: set moves each ball to its final position", () => {
    const axes = renderedAxes(`
      f = line(0,0,'marker','.');
      for r = 1:5
        x = r*2; y = r*r;
        set(f,'xdata',x,'ydata',y);
      end
    `);
    // The single line trace should reflect the final iteration (r=5).
    const line = axes.traces.find(t => t.marker === ".");
    expect(line).toBeDefined();
    expect(line!.x).toEqual([10]);
    expect(line!.y).toEqual([25]);
  });
});
