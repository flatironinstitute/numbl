export function allocFloat64Array(
  x: number | number[] | Float64Array
): Float64Array {
  if (typeof x === "number") {
    return new Float64Array(x);
  }
  return new Float64Array(x);
}
