%% Scalar comparisons return logical
x = 3 < 5;
assert(islogical(x));
assert(x == true);

y = 3 > 5;
assert(islogical(y));
assert(y == false);

%% Scalar equality returns logical
z = 3 == 3;
assert(islogical(z));

%% Logical AND of comparison results returns logical
a = (3 < 5) & (2 > 1);
assert(islogical(a));

%% NOT of comparison returns logical
b = ~(3 < 5);
assert(islogical(b));
assert(b == false);

%% Logical indexing with scalar comparison result
rk = NaN(1, 1);
index = false;
index = index & ~(abs(0.5 - 0.3) < 0.1);
rk(index) = 5;
assert(isnan(rk));

disp('SUCCESS')
