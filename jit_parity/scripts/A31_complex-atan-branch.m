% TEST: atan() of a pure-imaginary value with magnitude > 1 (on the branch cut).
% atan(2i) principal value = pi/2 + 0.54931i = 1.5708 + 0.54931i (C is correct).
%
% opt0 (interp): -1.5708 + 0.54931i
% opt1 (JS-JIT): -1.5708 + 0.54931i
% opt2 (C-JIT):   1.5708 + 0.54931i   <-- DIVERGES (real-part SIGN flips)
%
% DIVERGING MODE: opt2 vs opt0/opt1 (value differs, not just format).
% CAUSE: interpreter & JS mtoc2_catan compute atan(z) = (i/2)*log((1-iz)/(1+iz))
%   by forming ONE log of a quotient. When that quotient is a negative real
%   (here -3), the single-log path lands on the wrong branch, flipping the
%   real part to -pi/2. C99 catan() uses the correct principal branch.
% JIT ENGAGEMENT: confirmed (--dump-c uses catan; %!numbl:assert_jit c passes at opt2).
disp(atan(2i));
disp(atan(5i));
disp(atan(40i));
