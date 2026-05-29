% Regression: `round` inside a fused per-element chain must use MATLAB
% semantics (round half away from zero), not JS Math.round semantics
% (round half toward +Inf). The non-fused path already mapped round to a
% MATLAB-compatible helper; the fused scalar-emit path fell through to
% Math.round.

%!numbl:assert_jit
function [r1, r2, r3] = round_variants(a)
  % Three independent fused chains so whichever emitter kicks in gets
  % tested. The inputs are chosen to straddle the half-value ties that
  % MATLAB and JS disagree on.
  r1 = round(a);
  r2 = round(a + 0.0);      % trivially "fused" but exercises a chain with no op
  r3 = round(a) + 1.0;      % fused chain with round as a leaf
end

a = [-2.5; -1.5; -0.5; 0.5; 1.5; 2.5];

for k = 1:20
  [r1, r2, r3] = round_variants(a);
end

% MATLAB: round(-2.5) = -3, round(-1.5) = -2, round(-0.5) = -1,
%         round(0.5) = 1,   round(1.5) = 2,   round(2.5) = 3
expected = [-3.0; -2.0; -1.0; 1.0; 2.0; 3.0];

assert(isequal(r1, expected), 'round(a) result wrong');
assert(isequal(r2, expected), 'round(a + 0.0) result wrong');
assert(isequal(r3, expected + 1.0), 'round(a) + 1 result wrong');

disp('SUCCESS')
