% Regression: `1 .^ NaN` must be NaN on every --opt level. The interpreter
% and JS-JIT use Math.pow (NaN for |base|==1 with a non-finite exponent),
% but the C-JIT emitted bare C99 pow, which returns 1 there — so --opt 2
% silently disagreed. The fix routes the scalar real power and the tensor
% power kernels through a `mtoc2_pow_real` helper that matches Math.pow.
assert(isnan(1 .^ NaN), '1 .^ NaN should be NaN');
assert(isnan((-1) .^ NaN), '(-1) .^ NaN should be NaN');

% Ordinary powers are unaffected.
assert(2 .^ 3 == 8, '2 .^ 3');
assert((-2) .^ 2 == 4, '(-2) .^ 2');
assert(abs(9 .^ 0.5 - 3) < 1e-12, '9 .^ 0.5');
assert(5 .^ 0 == 1, '5 .^ 0');
assert(2 .^ -2 == 0.25, '2 .^ -2');

% Tensor power kernels route through the same helper (non-negative bases
% so the real-power kernel runs rather than the complex promotion path).
v = [1 1 2] .^ [NaN NaN 3];
assert(isnan(v(1)) && isnan(v(2)) && v(3) == 8, 'tensor power with NaN exponent');

disp('SUCCESS')
