% Test that sum works on scalar logical values

% Test 1: sum of scalar true
result1 = sum(true);
assert(result1 == 1, 'sum of true');

% Test 2: sum of scalar false
result2 = sum(false);
assert(result2 == 0, 'sum of false');

% Test 3: sum of logical scalar from comparison
x = 5;
result3 = sum(x > 3);
assert(result3 == 1, 'sum of comparison result');

% Test 4: The chebfun pattern - sum(abs(x) > 0) with scalar
result4 = sum(abs(-3) > 0);
assert(result4 == 1, 'sum of abs > 0 scalar');

disp('SUCCESS');
