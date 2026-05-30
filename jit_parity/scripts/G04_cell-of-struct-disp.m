% TEST: disp() of a cell containing a struct: c = {struct('a',1), 5}.
% opt0 (interp, reference): {    a: 1, 5}
% opt1 (JS-JIT): {[object Object], 5}        <-- DIVERGES (raw JS coercion)
% opt2 (C-JIT):  {    a: 1<newline>, 5}      <-- DIVERGES (stray newline)
% DIVERGING MODE: opt1 and opt2 (all three differ).
%
% Cause: the JIT cell-display helpers don't handle a struct slot the way
%   the interpreter does. JS mtoc2__format_cell_slot (runtime/cell/cell.js)
%   has no Struct branch and falls through to String(v) -> "[object
%   Object]". C's emitSlotInlineDisp (codegen/emitCellTypedef.ts) calls the
%   struct _disp whose per-field lines each end in '\n', leaving a stray
%   newline before the next slot.
% FIX DIRECTION: give both JIT cell-slot display paths a struct branch that
%   matches the interpreter's inline struct rendering.
% (Companion: a cell-valued field inside a struct -- G05 covers nesting.)
for i = 1:1
  c = {struct('a', 1), 5};
  disp(c);
end
