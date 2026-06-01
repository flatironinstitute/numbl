% TEST: disp() of a scalar double whose 5-significant-digit rendering is
% an exact tie. 1/256 = 0.00390625.
%   opt0/1: "0.0039063"   (mtoc2_format_double JS: toExponential(4),
%                          round-half-away)
%   opt2:   "0.0039062"   (mtoc2_format_double C: snprintf %.4e,
%                          round-half-to-even)
% DIVERGING MODES: opt2 vs opt0/opt1.
% CAUSE: format_double.js uses x.toExponential(4) (half-away); the C
%        sibling format_double.h uses snprintf("%.4e") (half-to-even).
% JIT ENGAGEMENT: confirmed (--dump-c non-empty; disp scalar path).
disp(1/256);
