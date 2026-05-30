% TEST: disp() of a cell containing a non-integer TENSOR slot.
% Builds {[n, n+0.5]} in a hot user function and disp's it.
%
% opt0 (interp):  {   3.1416   3.6416}
% opt1 (JS-JIT):  {   3.141592653589793   3.641592653589793}   <-- DIVERGES
% opt2 (C-JIT):   {   3.1416   3.6416}
% DIVERGING MODE: opt1 only (opt0==opt2).
%
% Cause: cell.js mtoc2__format_tensor() formats each element via
% mtoc2__format_scalar() which returns String(v) for non-integers
% (full JS precision) instead of the toPrecision(5) formatNumber rule.
% The C path's emitSlotInlineDisp tensor branch uses
% mtoc2_disp_tensor_inline (correct), so C matches the interpreter here.
% JIT engagement: CONFIRMED (assert_jit c passes; jsgen=1, cgen=1).
function out = f(n)
  %!numbl:assert_jit c
  c = {[n, n+0.5]};
  disp(c);
  out = n;
end
acc = 0;
for i = 1:1
  acc = acc + f(pi);
end
