% TEST: a function whose output is assigned only inside an `if` with no
% matching `else`, called so the condition is false.
% MATLAB / opt0 (interp): error "Output argument 'r' is not assigned".
% opt1 (JS-JIT): same (unassigned output marshals back undefined -> bail).
% opt2 (C-JIT, before fix): silently returns 0   <-- DIVERGED
% DIVERGING MODE: opt2 only.
%
% Cause: C predeclares the output to 0. FIX: the definite-assignment check
%   (definiteAssign.ts) declines C-JIT for a real function whose output may
%   be returned before assignment.
function r = g(c)
  if c
    r = 5;
  end
end
disp(g(false));
