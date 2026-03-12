% Test deal builtin function

% deal(A) - replicate single input to all outputs
[a, b, c] = deal(42);
assert(a == 42);
assert(b == 42);
assert(c == 42);

% deal(A1,...,An) - distribute inputs to outputs
[x, y, z] = deal(1, 2, 3);
assert(x == 1);
assert(y == 2);
assert(z == 3);

% deal with different types
[s, n] = deal('hello', 5);
assert(strcmp(s, 'hello'));
assert(n == 5);

% deal with single output
a = deal(99);
assert(a == 99);

% deal with arrays
[a, b] = deal([1 2 3], [4 5 6]);
assert(isequal(a, [1 2 3]));
assert(isequal(b, [4 5 6]));

% deal replicating an array
[a, b] = deal([1 2 3]);
assert(isequal(a, [1 2 3]));
assert(isequal(b, [1 2 3]));

disp('SUCCESS');
