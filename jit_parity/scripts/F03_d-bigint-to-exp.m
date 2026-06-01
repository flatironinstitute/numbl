% TEST: %d of a large integer-valued double (>= 1e18).
%   opt0/1: "1000000000000000000|10000000000000000000|100000000000000000000"
%           (JS: Number.isInteger true -> String(Math.abs(n)))
%   opt2:   "1.000000e+18|1.000000e+19|1.000000e+20"
%           (C emit_int requires fabs(raw) < 1e18 to treat as int; at/above
%            1e18 it falls back to the %e branch)
% DIVERGING MODES: opt2 vs opt0/opt1.
% CAUSE: C mtoc2__emit_int caps the integer fast-path at fabs<1e18, so big
%        integers print as %e; the JS engine prints the full decimal string.
% JIT ENGAGEMENT: confirmed (--dump-c non-empty).
fprintf('%d|%d|%d\n', 1e18, 1e19, 1e20);
