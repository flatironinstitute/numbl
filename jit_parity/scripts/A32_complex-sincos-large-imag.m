% TEST: sin()/cos() of a large PURE-imaginary value (overflow on the real lane).
%
% sin(720i):  opt0 NaN + Infinityi | opt1 NaN + Infinityi | opt2 Infinityi   <-- DIVERGES
% cos(720i):  opt0 Infinity + NaNi | opt1 Infinity + NaNi | opt2 Infinity     <-- DIVERGES
%
% DIVERGING MODE: opt2 vs opt0/opt1.
% CAUSE: JS/interp use the real decomposition
%   sin(z) = sin(re)*cosh(im) + i*cos(re)*sinh(im)
%   cos(z) = cos(re)*cosh(im) - i*sin(re)*sinh(im).
%   With re=0 and cosh(720)=Inf, the surviving lane is sin(0)*Inf = 0*Inf = NaN.
%   C99 csin/ccos handle the imaginary-axis special case (Annex G) cleanly.
% JIT ENGAGEMENT: confirmed (csin/ccos in --dump-c).
disp(sin(720i));
disp(cos(720i));
