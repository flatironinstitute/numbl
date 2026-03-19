% Test 'omitnan' flag for reduction builtins

x = [1, NaN, 3, 4, NaN];

% sum
assert(isnan(sum(x)));
assert(sum(x, 'omitnan') == 8);
assert(sum(x, 'includenan') ~= sum(x, 'includenan') || isnan(sum(x, 'includenan')));

% prod
assert(isnan(prod(x)));
assert(prod(x, 'omitnan') == 12);

% mean
assert(isnan(mean(x)));
assert(mean(x, 'omitnan') == 8/3);

% std
assert(isnan(std(x)));
s = std(x, 0, 'omitnan');
expected_std = std([1, 3, 4]);
assert(abs(s - expected_std) < 1e-10);

% var
assert(isnan(var(x)));
v = var(x, 0, 'omitnan');
expected_var = var([1, 3, 4]);
assert(abs(v - expected_var) < 1e-10);

% median
assert(median(x, 'omitnan') == 3);

% Test with matrix along dimension
M = [1, NaN, 3; NaN, 5, 6];
s = sum(M, 1, 'omitnan');
assert(s(1) == 1);
assert(s(2) == 5);
assert(s(3) == 9);

m = mean(M, 1, 'omitnan');
assert(m(1) == 1);
assert(m(2) == 5);
assert(m(3) == 4.5);

% Test all-NaN slice gives NaN for mean
allnan = [NaN, NaN, NaN];
assert(isnan(mean(allnan, 'omitnan')));

% Test sum of all-NaN gives 0 (MATLAB behavior)
assert(sum(allnan, 'omitnan') == 0);

% Test 'all' with omitnan
M2 = [1, NaN; NaN, 4];
assert(sum(M2, 'all', 'omitnan') == 5);

disp('SUCCESS');
