% Regression: a power with a non-finite exponent whose Math.pow result is
% NaN (1 .^ Inf, 1 ^ Inf, 1 .^ -Inf) must NOT stack-overflow the interpreter.
%
% In binop, the numeric fast path computed Math.pow(1, Inf) = NaN and
% `break`ed out of the switch to reach the RuntimeValue slow path (where
% complex/NaN results are handled). But the *secondary* fast path then
% re-called binop(op, 1, Inf) with the same plain numbers, which hit the
% same break and recursed forever ("Maximum call stack size exceeded"). The
% fix skips the secondary fast path when both operands are already plain
% numbers, so evaluation falls through to the slow path and returns NaN.
assert(isnan(1 .^ Inf), '1 .^ Inf must be NaN (not crash)');
assert(isnan(1 ^ Inf), '1 ^ Inf must be NaN (not crash)');
assert(isnan(1 .^ -Inf), '1 .^ -Inf must be NaN (not crash)');
assert(isnan(1 .^ NaN), '1 .^ NaN must be NaN');

% Negative base with a fractional exponent still promotes to complex.
assert(~isreal((-2) ^ 0.5), '(-2)^0.5 should be complex');
assert(abs(imag((-2) ^ 0.5) - sqrt(2)) < 1e-12, '(-2)^0.5 value');

% Boolean operands still convert to numbers through the secondary path.
assert(true .^ 2 == 1, 'true .^ 2');
assert(2 .^ 3 == 8, '2 .^ 3');

disp('SUCCESS')
