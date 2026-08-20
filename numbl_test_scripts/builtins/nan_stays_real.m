% A NaN input is not an out-of-domain input.
%
% The maybe-complex element-wise builtins (sqrt, log, asin, ...) switch to
% their complex branch when the real branch returns NaN, which is how
% sqrt(-4) becomes 2i. A NaN *input* also makes the real branch return NaN,
% and treating that as out-of-domain made sqrt(NaN) a complex NaN — whose
% real part numbl then reports as 0, so `real(sqrt(NaN))` was 0 rather than
% NaN, silently turning a NaN tail of an array into zeros. MATLAB returns a
% real NaN for all of these.

assert(isnan(sqrt(NaN)) && isreal(sqrt(NaN)), 'sqrt(NaN)');
assert(isnan(log(NaN)) && isreal(log(NaN)), 'log(NaN)');
assert(isnan(log2(NaN)) && isreal(log2(NaN)), 'log2(NaN)');
assert(isnan(log10(NaN)) && isreal(log10(NaN)), 'log10(NaN)');
assert(isnan(asin(NaN)) && isreal(asin(NaN)), 'asin(NaN)');
assert(isnan(acos(NaN)) && isreal(acos(NaN)), 'acos(NaN)');
assert(isnan(real(sqrt(NaN))), 'real(sqrt(NaN)) is NaN, not 0');

% Out-of-domain inputs still go complex.
assert(abs(sqrt(-4) - 2i) < 1e-15, 'sqrt(-4)');
assert(abs(log(-1) - pi*1i) < 1e-15, 'log(-1)');
assert(real(sqrt(-Inf)) == 0 && imag(sqrt(-Inf)) == Inf, 'sqrt(-Inf)');

% Mixed array: the NaN slot stays NaN while the negative slot goes complex.
v = sqrt([1 NaN -4 0]);
assert(v(1) == 1, 'array in-domain');
assert(isnan(v(2)) && imag(v(2)) == 0, 'array NaN slot stays a real NaN');
assert(abs(v(3) - 2i) < 1e-15, 'array out-of-domain slot');
assert(v(4) == 0, 'array zero slot');

% An all-NaN array must not be promoted to complex at all.
w = sqrt([NaN NaN]);
assert(isreal(w) && all(isnan(w)), 'all-NaN array stays real');

disp('SUCCESS')
