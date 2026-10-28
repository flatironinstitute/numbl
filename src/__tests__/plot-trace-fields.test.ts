import { describe, it, expect } from "vitest";
import {
  TRACE_FIELDS,
  TRACE_FIELD_NAMES,
} from "../numbl-core/plotting/axesTraces.js";
import { createDefaultAxes } from "../numbl-core/plotting/figuresReducer.js";
import {
  GRID_FIELDS,
  KIND_TO_CATEGORY,
  exportKind,
} from "../graphics/figureHdf5Schema.js";
import { CANVAS_RENDER_ORDER } from "../graphics/renderers/index.js";

// The per-field table in axesTraces.ts drives the reducer, the axis limits,
// the 2-D/3-D routing and the HDF5 layout. These checks catch a trace field
// that one of those consumers would silently drop.

describe("trace field table", () => {
  it("maps every field to an HDF5 kind that reads back into it", () => {
    for (const field of TRACE_FIELD_NAMES) {
      const kind = exportKind(field);
      expect(KIND_TO_CATEGORY[kind]?.field, kind).toBe(field);
      expect(!!KIND_TO_CATEGORY[kind].single, field).toBe(
        TRACE_FIELDS[field].multiplicity === "one"
      );
    }
  });

  it("uses each HDF5 kind for one field only", () => {
    const kinds = TRACE_FIELD_NAMES.flatMap(f => [...TRACE_FIELDS[f].kinds]);
    expect(new Set(kinds).size).toBe(kinds.length);
    for (const kind of Object.keys(GRID_FIELDS))
      expect(KIND_TO_CATEGORY[kind], kind).toBeDefined();
  });

  it("gives every field an empty default", () => {
    const axes = createDefaultAxes();
    for (const field of TRACE_FIELD_NAMES) {
      if (TRACE_FIELDS[field].multiplicity === "many")
        expect(axes[field], field).toEqual([]);
      else expect(axes[field], field).toBeUndefined();
    }
    // Fresh arrays each time (the HDF5 reader pushes into them).
    expect(createDefaultAxes().traces).not.toBe(axes.traces);
  });

  it("has a 2-D canvas renderer for every field that is not 3-D-only", () => {
    const threeDOnly = TRACE_FIELD_NAMES.filter(f => {
      const spec = TRACE_FIELDS[f];
      const sample = spec.multiplicity === "many" ? [{}] : {};
      return (spec.is3D as (v: unknown) => boolean)(sample);
    });
    expect(threeDOnly.sort()).toEqual(
      [
        "bar3Traces",
        "bar3hTraces",
        "plot3Traces",
        "quiver3Traces",
        "surfTraces",
      ].sort()
    );
    expect([...CANVAS_RENDER_ORDER].sort()).toEqual(
      TRACE_FIELD_NAMES.filter(f => !threeDOnly.includes(f)).sort()
    );
  });
});
