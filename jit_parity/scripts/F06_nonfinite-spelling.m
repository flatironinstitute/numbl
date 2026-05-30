% AREA: C-JIT printf formatting — Inf / NaN spelling.
%
% WHAT IT TESTS: how Inf / -Inf / NaN render under %f %e %g and %d.
% The interpreter and JS-JIT spell them via format_engine.js numStr()
% ("Inf"/"-Inf" for the float specs) and via toExponential for the %d
% fallback ("Infinity"/"NaN"); the C-JIT emits libc spellings.
%
%   fprintf('[%f][%e][%g]', 1/0,1/0,1/0)
%     opt0/opt1: [Inf][Inf][Inf]
%     opt2:      [Infinity][Infinity][Infinity]      <-- DIVERGES
%   fprintf('[%f][%g]', -1/0,-1/0)
%     opt0/opt1: [-Inf][-Inf]   opt2: [-Infinity][-Infinity]
%   fprintf('[%d][%d]', 1/0, 0/0)
%     opt0/opt1: [Infinity][NaN]   opt2: [inf][-nan]   <-- DIVERGES
%
% DIVERGING MODE: opt2 only (opt0 == opt1).
%
% CAUSE: format_engine.h mtoc2__emit_float hardcodes "Infinity"/"-Infinity"
% (JS numStr uses "Inf"/"-Inf"), and the %d-of-non-finite path falls into
% the %e branch which in C emits libc "inf"/"-nan" instead of the JS
% toExponential strings "Infinity"/"NaN". C must match the JS spellings.
% (disp() of Inf already agrees across modes; only the sprintf/fprintf
% numStr path diverges. The same C %g slot path makes disp({1/0}) print
% {inf} at opt2 vs {Infinity} elsewhere.)
%
% JIT ENGAGEMENT: top-level fprintf -> whole-scope JIT'd (dump-c non-empty).
fprintf('[%f][%e][%g]\n', 1/0, 1/0, 1/0);
fprintf('[%f][%g]\n', -1/0, -1/0);
fprintf('[%d][%d]\n', 1/0, 0/0);
