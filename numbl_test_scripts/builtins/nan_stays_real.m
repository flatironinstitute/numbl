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

% Under the JIT the same must hold. When the compiler cannot prove an
% argument is non-negative it lifts the call onto the complex path, and a
% NaN lifted that way used to come back as 0 + NaN*i — a different answer
% from the interpreter for `real`, `isreal`, `jsonencode` and display.
x = [4 NaN -4];
re = zeros(1, 3);
im = zeros(1, 3);
for k = 1:3
    %!numbl:assert_jit
    y = sqrt(x(k) + 0);
    re(k) = real(y);
    im(k) = imag(y);
end
assert(re(1) == 2 && im(1) == 0, 'jit sqrt(4)');
assert(isnan(re(2)) && im(2) == 0, 'jit sqrt(NaN) keeps NaN on the real lane');
assert(re(3) == 0 && im(3) == 2, 'jit sqrt(-4)');

lre = zeros(1, 2);
lim = zeros(1, 2);
xs = [exp(1) NaN];
for k = 1:2
    %!numbl:assert_jit
    y = log(xs(k) + 0);
    lre(k) = real(y);
    lim(k) = imag(y);
end
assert(abs(lre(1) - 1) < 1e-15 && lim(1) == 0, 'jit log(e)');
assert(isnan(lre(2)) && lim(2) == 0, 'jit log(NaN)');

disp('SUCCESS')
