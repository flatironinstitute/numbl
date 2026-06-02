% TEST: a variable assigned ONLY inside a for-loop body, read after the
% loop, when the loop runs zero times (empty range). The realistic shape is
%   for i = 1:numel(items); r = process(items(i)); end; use(r)
% with an empty `items`.
% MATLAB / opt0 (interp): error "Undefined function or variable 'x'".
% opt1 (JS-JIT): same error (the unassigned local marshals back as
%   undefined and bails to the interpreter).
% opt2 (C-JIT, before fix): silently returns 0   <-- DIVERGED
% DIVERGING MODE: opt2 only (silent 0 / corruption).
%
% Cause: C predeclares every local at function top (`double x = 0.0;`), so a
%   read on a path where the assignment never ran leaks the 0 default. FIX: a
%   definite-assignment check (definiteAssign.ts) declines C-JIT when a local
%   may be read before assignment, so the interpreter reproduces MATLAB's
%   undefined-variable error. All three modes now agree (error -> <ERROR>).
function r = f(n)
  for k = 1:n
    x = k * 10;
  end
  r = x;
end
disp(f(0));
