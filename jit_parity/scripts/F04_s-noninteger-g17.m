% TEST: %s given a numeric (non-text) argument that is a non-integer.
%   sprintf/fprintf %s of 0.1
%   opt0/1: "0.1"                  (JS toString -> String(0.1))
%   opt2:   "0.10000000000000001"  (C mtoc2__num_to_str -> snprintf %.17g)
% DIVERGING MODES: opt2 vs opt0/opt1.
% CAUSE: numbl's %s on a number routes through toString()==String(n) in
%        JS/interpreter, but the C engine renders non-integers via
%        snprintf("%.17g", x), exposing the full double round-trip digits.
% JIT ENGAGEMENT: confirmed (--dump-c non-empty).
fprintf('[%s]\n', 0.1);
