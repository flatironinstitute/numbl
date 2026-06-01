% TEST: disp() of a cell containing a non-integer scalar.
% Builds a tuple cell {n, n+1/3} in a hot user function (engages
% cjit-call at --opt 2 and jit-call at --opt 1) and disp's it.
%
% opt0 (interp):  {3.1416, 3.4749}                          (formatNumber: toPrecision(5))
% opt1 (JS-JIT):  {3.141592653589793, 3.4749259869231266}   (String(v): full JS precision)
% opt2 (C-JIT):   {3.14159, 3.47493}                         (printf("%g"): 6 sig digits)
% ALL THREE DIVERGE.
%
% Cause: cell-slot scalar disp bypasses the proper formatNumber-equivalent.
%  - JS:  cell.js mtoc2__format_scalar() returns String(v) for non-integers.
%  - C:   emitCellTypedef.ts emitSlotInlineDisp() emits printf("%g", expr)
%         instead of mtoc2_disp_double().
% (Struct field disp uses mtoc2_disp_double and matches — see struct_num.m.)
% JIT engagement: CONFIRMED (assert_jit c passes; jsgen=1, cgen=1).
function out = f(n)
  %!numbl:assert_jit c
  c = {n, n + 1/3};
  disp(c);
  out = n;
end
acc = 0;
for i = 1:1
  acc = acc + f(pi);
end
