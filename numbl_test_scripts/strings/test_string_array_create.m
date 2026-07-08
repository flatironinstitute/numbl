% Test string array creation and shape queries

% concatenating string scalars builds arrays
s = ["a" "bb" "ccc"];
assert(isstring(s));
assert(strcmp(class(s), 'string'));
assert(isequal(size(s), [1 3]));
assert(numel(s) == 3);

% 2-D construction
str = ["Mercury" "Gemini" "Apollo"; "Skylab" "Skylab B" "ISS"];
assert(isequal(size(str), [2 3]));
assert(str(2, 2) == "Skylab B");

% string([]) is a 0x0 empty string array; empties vanish in concatenation
e = string([]);
assert(isstring(e));
assert(isempty(e));
assert(isequal(size(e), [0 0]));
out = [string([]), string('gmsh'), string('abc')];
assert(isequal(size(out), [1 2]));
assert(out(1) == "gmsh");

% char operands become string elements (string wins over char)
m = ["ab" 'cd'];
assert(isstring(m));
assert(numel(m) == 2);
assert(m(2) == "cd");
m2 = ['ab' "cd"];
assert(numel(m2) == 2);

% numbers and logicals convert elementwise
x = ["a", 1.5, pi, true];
assert(numel(x) == 4);
assert(x(2) == "1.5");
assert(x(3) == "3.1416");
assert(x(4) == "true");

% vertical concatenation
v = ["a"; "b"; "c"];
assert(isequal(size(v), [3 1]));
assert(v(2) == "b");

% transpose
t = ["a" "b"]';
assert(isequal(size(t), [2 1]));
assert(t(2) == "b");

% reshape (column-major order preserved)
r = reshape(["a" "b" "c" "d"], 2, 2);
assert(isequal(size(r), [2 2]));
assert(r(2, 1) == "b");
assert(r(1, 2) == "c");

% a 1x1 result is a string scalar
one = ["only"];
assert(isequal(size(one), [1 1]));
assert(one == "only");

% strings() builder
g = strings(2, 3);
assert(isequal(size(g), [2 3]));
assert(strlength(g(1, 1)) == 0);
g0 = strings(0);
assert(isempty(g0));
assert(isstring(g0));

disp('SUCCESS')
