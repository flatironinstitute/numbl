% mod()/rem() with Inf, NaN, and large divisors.
% MATLAB uses mod(a,b) = a - b*floor(a/b), with special rules for Inf
% divisors, and avoids the precision loss that the naive
% ((a%b)+b)%b formula has for very large |b|.

% Inf divisor: result has the sign of b (or equals a when signs match).
assert(mod(3,  Inf)  == 3);
assert(mod(-3, Inf)  == Inf);
assert(mod(3, -Inf)  == -Inf);
assert(mod(-3,-Inf)  == -3);
assert(mod(0,  Inf)  == 0);
assert(mod(0, -Inf)  == 0);

% Non-finite dividend: NaN.
assert(isnan(mod(Inf, 3)));
assert(isnan(mod(-Inf, 3)));
assert(isnan(mod(NaN, 3)));
assert(isnan(mod(3, NaN)));

% Large divisor: naive ((a%b)+b)%b loses `a` in float add. MATLAB returns a.
assert(mod(3, 1e18)  == 3);
assert(mod(-3, 1e18) == 1e18 - 3);  % wraps into [0, 1e18)

% rem semantics: same sign as dividend, NaN for 0 or Inf dividend.
assert(rem(3,  Inf)  == 3);
assert(rem(-3, Inf)  == -3);
assert(rem(3, -Inf)  == 3);
assert(isnan(rem(Inf, 3)));
assert(isnan(rem(3, 0)));

disp('SUCCESS');
