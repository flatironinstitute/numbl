% TEST: disp() of a struct with a NESTED struct field and a cell field:
%   s = struct('p', struct('a',1,'b',2), 'q', 9)
% opt0 (interp, reference): inlines the nested struct right after "p: "
%   (no extra newline) -> "    p:     a: 1 / b: 2 / q: 9".
% opt1 (JS-JIT): inserts a stray newline after "p:" before the nested body;
%   a cell-valued field additionally leaks the raw cell wrapper internals
%   (mtoc2Tag/shape/data...)   <-- DIVERGES
% opt2 (C-JIT):  inserts the same stray newline after "p:"  <-- DIVERGES
% DIVERGING MODE: opt1 and opt2.
%
% Cause: C struct _disp (emitNamedTypedef.ts) emits "    p: " then a forced
%   '\n' before recursing; JS disp_struct.js does the same and, for a
%   cell-valued field, has no cell branch so it recurses over the cell
%   wrapper's Object.keys.
% FIX DIRECTION: match the interpreter -- inline a nested struct after the
%   "field: " label (no forced newline), and render a cell-valued field via
%   the proper cell formatter.
for i = 1:1
  s = struct('p', struct('a',1,'b',2), 'q', 9);
  disp(s);
end
