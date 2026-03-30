% Test str2double with cell array inputs

% Cell array of numeric strings
X = str2double({'1.5', '2.7', '3.14'});
assert(isequal(size(X), [1 3]));
assert(abs(X(1) - 1.5) < 1e-12);
assert(abs(X(2) - 2.7) < 1e-12);
assert(abs(X(3) - 3.14) < 1e-12);

% Cell array with non-convertible strings returns NaN
X = str2double({'42', 'hello', '3.14'});
assert(X(1) == 42);
assert(isnan(X(2)));
assert(abs(X(3) - 3.14) < 1e-12);

% Thousands separator (commas)
assert(str2double('1,000') == 1000);
assert(str2double('1,234,567.89') == 1234567.89);

% Scientific notation
assert(str2double('1.5e3') == 1500);
assert(str2double('-2.5E-4') == -2.5e-4);

% Leading +
assert(str2double('+42') == 42);

disp('SUCCESS');
