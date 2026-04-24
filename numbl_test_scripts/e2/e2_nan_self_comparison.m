% e2 NaN self-comparison: x ~= x and x == x must give correct IEEE 754
% results under -ffast-math / -ffinite-math-only.
%
% The compiler flag -ffinite-math-only (implied by -ffast-math, which
% e2 kernels use for vectorisation) lets the compiler assume no NaN/Inf
% values, causing it to constant-fold `x != x` to 0 and `x == x` to 1.
% The e2 emitter works around this by replacing self-comparisons with a
% bit-pattern NaN check (numbl_is_nan_fp) that is opaque to the
% optimiser.

n = 6000;
x = linspace(-2, 2, n);
x(1001) = NaN;
x(3000) = NaN;

% ~= self-comparison: NaN detection idiom
nan_mask = x ~= x;
assert(islogical(nan_mask), 'nan_mask should be logical');
assert(sum(nan_mask) == 2, sprintf('expected 2 NaNs, found %d', sum(nan_mask)));
clean = x(~nan_mask);
assert(~any(isnan(clean)), 'clean should have no NaNs');

% == self-comparison: should be 1 for finite, 0 for NaN
eq_mask = x == x;
assert(islogical(eq_mask), 'eq_mask should be logical');
assert(sum(~eq_mask) == 2, sprintf('expected 2 non-equal (NaN), found %d', sum(~eq_mask)));

% Normal (non-self) comparisons must still work correctly through the
% NaN values: NaN > anything = false, NaN < anything = false.
y = x;
y(1001) = 5.0;   % replace the first NaN with a finite value
cross_mask = x > y;
% At positions where x was NaN, x > y should be false
assert(~cross_mask(1001), 'NaN > finite should be false');
assert(~cross_mask(3000), 'NaN > NaN should be false');

disp('SUCCESS')
