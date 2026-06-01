% TEST: exp() of a COMPLEX-typed value whose imaginary part is 0 and whose
% real part overflows. Here the JS-JIT (opt1) is the odd one out.
%
% exp(720+0i): opt0 Infinity | opt1 Infinity + NaNi | opt2 Infinity
%                                       ^^^^ opt1 DIVERGES from opt0 AND opt2
% Also reproduces with a COMPUTED complex value:  z = 720+1i; z = z - 1i; exp(z)
%
% DIVERGING MODE: opt1 (JS-JIT) vs opt0/opt2.
% CAUSE: mtoc2_cexp = exp(re)*cos(im) + i*exp(re)*sin(im). exp(720)=Inf, and the
%   imag lane is Inf*sin(0)=Inf*0=NaN. The interpreter special-cases the real
%   axis and C99 cexp handles im=0 per Annex G, both giving clean Infinity.
% JIT ENGAGEMENT: confirmed (cexp in --dump-c, c_bytes ~10.5k).
z = 720 + 1i;
z = z - 1i;
disp(exp(z));
disp(exp(720 + 0*1i));
