% AREA: C-JIT printf formatting — half-way rounding direction.
%
% WHAT IT TESTS: %f / %e / %g of exact half-way values.
% The interpreter and JS-JIT round half AWAY from zero (JS toFixed /
% toExponential / toPrecision); the C-JIT defers to libc snprintf which
% rounds half to EVEN (banker's rounding).
%
%   fprintf('%.0f|%.0f|%.0f|%.0f|%.2f', 0.5,1.5,2.5,3.5,0.125)
%     opt0/opt1: 1|2|3|4|0.13
%     opt2:      0|2|2|4|0.12   <-- DIVERGES (round-half-to-even)
%   fprintf('%.0e|%.0e', 2.5,1.5)  opt0/1: 3e+00|2e+00  opt2: 2e+00|2e+00
%   fprintf('%.1g|%.2g', 2.5,0.125) opt0/1: 3|0.13     opt2: 2|0.12
%
% DIVERGING MODE: opt2 only (opt0 == opt1).
%
% CAUSE: src/numbl-core/jit/builtins/runtime/io/format_engine.h
% mtoc2__emit_float defers %f/%e/%g to snprintf (half-to-even); the JS
% sibling format_engine.js uses toFixed/toExponential/toPrecision
% (half-away). The reference interpreter (helpers/string.ts sprintfFormat)
% is half-away. The C path must round half-away to match.
%
% JIT ENGAGEMENT: top-level fprintf is a void call -> whole-scope JIT'd
% (dump-c non-empty; %!numbl:assert_jit c passes).
fprintf('%.0f|%.0f|%.0f|%.0f|%.2f\n', 0.5, 1.5, 2.5, 3.5, 0.125);
fprintf('%.0e|%.0e\n', 2.5, 1.5);
fprintf('%.1g|%.2g\n', 2.5, 0.125);
