% TEST: function local assigned only on a not-taken branch, then read in arithmetic.
%   `y` is assigned only when x>0; f(-2) reads `y` on the path where it was never set.
% opt0 (interp): ERROR "Undefined function or variable 'y'"   (correct MATLAB semantics)
% opt1 (JS-JIT): prints  6  then  NaN   (JS `let y;` => undefined; undefined+1 = NaN)
% opt2 (C-JIT):  prints  6  then  1     (C `double y = 0.0;`  => 0+1 = 1)
% DIVERGING MODES: BOTH opt1 and opt2 (three-way: error vs NaN vs 1).
% JIT ENGAGEMENT: confirmed (jsgen=1 cgen=1 via --dump-js/--dump-c).
% HYPOTHESIZED CAUSE: the JIT predeclares every function local (C: defaultInitFor()=0.0;
%   JS: bare `let`), so reading a variable on a path where it was never assigned silently
%   yields 0 (C) / undefined->NaN (JS) instead of raising the interpreter's
%   "Undefined variable" error. emitStmt.ts collectAllLocals + emitFunction predeclare.
function r = f(x)
  if x > 0
    y = 5;
  end
  r = y + 1;
end
disp(f(3));
disp(f(-2));
