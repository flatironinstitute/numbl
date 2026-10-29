/** The MATLAB class of a tensor's elements (what `class` reports). Every
 *  class stores its elements in a Float64Array; the class fixes their
 *  semantics (a logical tensor holds 0/1 and is real). Kept in its own
 *  module so that the GPU device types can use it without importing
 *  types.ts. */
export type ElementClass = "double" | "single" | "logical";
