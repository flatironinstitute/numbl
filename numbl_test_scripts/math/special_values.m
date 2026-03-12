% Test special values: NaN, Inf, isnan, isinf, isempty

% NaN
n = NaN;
assert(isnan(n));
assert(~isnan(3.14));

% Inf
inf_val = Inf;
assert(isinf(inf_val));
assert(~isinf(3.14));
assert(inf_val > 1e300);

% -Inf
ninf = -Inf;
assert(isinf(ninf));
assert(ninf < -1e300);

% isempty
assert(isempty([]));
assert(~isempty([1, 2, 3]));
assert(isempty(''));
assert(~isempty('hello'));

% isfinite
assert(isfinite(3.14));
assert(~isfinite(Inf));
assert(~isfinite(-Inf));
assert(~isfinite(NaN));

% NaN arithmetic
assert(isnan(NaN + 1));
assert(isnan(NaN * 0));
assert(isnan(0 / 0));

% Inf arithmetic
assert(isinf(1 / 0));
assert(isinf(Inf + 1));

disp('SUCCESS')
