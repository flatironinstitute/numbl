% Test randperm builtin

%% Basic: randperm(n) returns permutation of 1:n
v = randperm(5);
assert(length(v) == 5);
assert(all(sort(v) == [1, 2, 3, 4, 5]));

%% randperm(n, k) returns k elements from 1:n without replacement
v = randperm(10, 3);
assert(length(v) == 3);
assert(all(v >= 1 & v <= 10));
% all unique
assert(length(unique(v)) == 3);

%% randperm(1) == [1]
v = randperm(1);
assert(v == 1);

%% randperm(n, n) is a full permutation
v = randperm(6, 6);
assert(length(v) == 6);
assert(all(sort(v) == 1:6));

%% randperm(n, 1) returns a single element
v = randperm(100, 1);
assert(length(v) == 1);
assert(v >= 1 && v <= 100);

%% Output is a row vector
v = randperm(5);
assert(size(v, 1) == 1);
assert(size(v, 2) == 5);

disp('SUCCESS');
