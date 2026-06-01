% TEST: multi-return user function whose 2nd output is assigned only on a not-taken branch,
%   but the caller requests it: `[p,q] = f(5)` with `b` set only when x>100.
% opt0 (interp): ERROR "Output argument 'b' (and maybe others) not assigned during call to 'f'" (correct)
% opt1 (JS-JIT): SAME error (JS leaves b undefined; the boundary detects it / bails to interp)
% opt2 (C-JIT):  prints "5 0"   (C predeclares output `b` to 0.0 and returns it via sret)
% DIVERGING MODE: opt2 only (C-JIT). opt1 matches opt0.
% JIT ENGAGEMENT: confirmed (jsgen=1 cgen=1).
% HYPOTHESIZED CAUSE: emitFunction predeclares each declared output slot
%   (`<T> cOut = 0.0;` / `_empty()`), so a never-assigned output is silently returned as 0
%   instead of triggering numbl's "Output argument not assigned" error. Same predeclaration
%   root cause as 01-03, on the output-slot path.
function [a, b] = f(x)
  a = x;
  if x > 100
    b = x;
  end
end
[p, q] = f(5);
fprintf('%d %d\n', p, q);
