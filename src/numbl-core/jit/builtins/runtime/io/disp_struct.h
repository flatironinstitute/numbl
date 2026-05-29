/* JS-only stub. The C path generates per-typedef `<name>_disp`
 * functions via `emitNamedTypedef`, so it never activates this
 * snippet; the JS path uses the `.js` sibling for a generic
 * runtime-walking struct disp helper. Keeping the empty `.h` here
 * lets the snippet registry resolve `mtoc2_disp_struct` to a real
 * registration that carries the paired `.js` body.
 */
