% TEST: sqrt() of a pure-real NEGATIVE scalar whose sign isn't statically
% known (so the JIT routes it through the complex-sqrt runtime helper).
% sqrt(-Inf) and sqrt(-1e308) must give 0 + (+/-)Inf*i and 0 + 1e154*i.
% opt0 (interp): re(z1)==0, isinf(imag(z1)), z2 = 0 + 1e154i, z3 = 0 + 2i
% opt1 (JS-JIT): WRONG — re(z1)=NaN, imag(z2)=Inf (instead of 1e154)
% opt2 (C-JIT):  correct (libm csqrt special-cases the real axis)
% DIVERGES: opt1 vs opt0/opt2 (re(z1) and imag(z2) columns).
% CAUSE: the JS complex-sqrt helper mtoc2_csqrt (cscalar.js) used Smith's
%   formula unconditionally; for a pure-real input it cancels/overflows:
%   sqrt(-Inf) -> sqrt((Inf + -Inf)/2) = sqrt(NaN) = NaN, and
%   sqrt(-1e308) -> sqrt((1e308 - -1e308)/2) overflows to Inf. The
%   interpreter (complexSqrt in math.ts) and C's libm csqrt both special-
%   case z.im == 0. Fixed by adding the same pure-real branch to the JS
%   mtoc2_csqrt. The result is printed via finite values + isinf() so the
%   Inf-spelling divergence (F06, deliberately excluded) doesn't leak in.
% JIT-ENGAGEMENT: all-fprintf void script -> whole scope JIT-compiles.
function y = f(x)
  y = sqrt(x);
end
z1 = f(-Inf);
z2 = f(-1e308);
z3 = f(-4);
fprintf('%d %d %g %g %g %g\n', real(z1) == 0, isinf(imag(z1)), ...
        real(z2), imag(z2), real(z3), imag(z3));
