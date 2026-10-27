% TEST: complex multiply whose real part a*c - b*d cancels exactly.
% With z = (1+2^-30) + i and w = (1-2^-30) + i, a*c = 1 - 2^-60 rounds to 1,
% so the separately rounded a*c - b*d is 0. A compiler that contracts it
% into fma(a, c, -b*d) keeps the exact product and gives -2^-60.
% DIVERGING MODE: opt0 (native addon built with gcc's default
%   -ffp-contract=fast for C) gave -8.6736173798840355e-19; opt1 (JS) and
%   opt2 (C-JIT, -std=c11) gave 0. clang contracts by default even with
%   -std=c11, so on macOS the C-JIT diverged as well.
% FIX: compile the addon and the C-JIT with -ffp-contract=off (and
%   `#pragma STDC FP_CONTRACT OFF` in the emitted C), so no mode fuses.
function p = f(z, w)
  p = z .* w;
end
z = [1 + 2^-30 + 1i, 2 + 1i];
w = [1 - 2^-30 + 1i, 3 - 1i];
for k = 1:300
  p = f(z, w);
end
fprintf('%.17g %.17g\n', real(p(1)), imag(p(1)));
