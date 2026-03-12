% Test legacy rand('seed', s) and randn('seed', s) syntax

% rand('seed', s) should seed and produce reproducible results
rand('seed', 42);
a1 = rand(1, 3);
rand('seed', 42);
a2 = rand(1, 3);
assert(all(a1 == a2));

% randn('seed', s) should seed and produce reproducible results
randn('seed', 7);
b1 = randn(1, 3);
randn('seed', 7);
b2 = randn(1, 3);
assert(all(b1 == b2));

% Different seeds should produce different results
rand('seed', 1);
c1 = rand(1, 5);
rand('seed', 2);
c2 = rand(1, 5);
assert(~all(c1 == c2));

disp('SUCCESS');
