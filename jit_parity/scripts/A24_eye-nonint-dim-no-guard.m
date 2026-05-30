% AREA: shape/structural builtins — eye() dynamic non-integer dim
%
% WHAT IT TESTS: eye(n, 3) with a runtime non-integer dim (2.6, read
% from a tensor element so it is not folded). The interpreter REJECTS a
% non-integer size; the JIT silently TRUNCATES and builds the matrix.
%
% f returns numel(eye(n, 3)) with n=2.6:
%   opt0 (interp, REFERENCE): ERROR "Size inputs must be nonnegative integers."
%   opt1 (JS-JIT):            6   (Math.trunc(2.6)=2 -> eye(2,3))   <-- DIVERGES
%   opt2 (C-JIT):             6   ((long)2.6=2 -> eye(2,3))         <-- DIVERGES
%
% HYPOTHESIZED CAUSE: src/numbl-core/jit/builtins/defs/shape/eye.ts emits
% raw `Math.trunc(arg)` / `(long)(arg)` for dynamic dims, with NO runtime
% validation. zeros/ones were fixed to route dynamic dims through
% `mtoc2_check_dim` (errors on non-integer, _construct.ts CHECK_DIM_SNIPPET);
% eye was not given the same guard, so it diverges from the interpreter's
% validateDim (array-construction.ts).
%
% JIT ENGAGEMENT: CONFIRMED. opt2 --dump-c call site is
%   mtoc2_eye_rect((long)(n), 3L)  (no mtoc2_check_dim wrapper).
function s = f(vals, k)
  n = vals(k);
  m = eye(n, 3);
  s = numel(m);
end
vals = [2.6 2.6 2.6 2.6 2.6];
total = -1;
for k = 1:5
  total = f(vals, k);
end
disp(total);
