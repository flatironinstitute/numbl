% TEST: atan() of a COMPLEX-typed value with zero imaginary part. JS-JIT (opt1)
% adds a spurious imaginary residue (or NaN at huge magnitude).
%
% atan(40+0i):    opt0 1.5458   | opt1 1.5458 - 5.5511e-17i | opt2 1.5458
% atan(1e300+0i): opt0 1.5708   | opt1 NaN + NaNi           | opt2 1.5708
%                                       ^^^^ opt1 DIVERGES from opt0 AND opt2
% Also reproduces with computed: z = 40+1i; z = z - 1i; atan(z).
%
% DIVERGING MODE: opt1 (JS-JIT) vs opt0/opt2.
% CAUSE: mtoc2_catan runs the complex-log formula even for a real input,
%   leaving a tiny imaginary residue; for huge real input the intermediate
%   1+iz overflows -> NaN. The interpreter and C99 catan return a clean real.
% JIT ENGAGEMENT: confirmed (catan in --dump-c).
z = 40 + 1i;
z = z - 1i;
disp(atan(z));
disp(atan(1e300 + 0*1i));
