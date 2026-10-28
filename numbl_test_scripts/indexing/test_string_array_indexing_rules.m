% String arrays: reads, stores, growth and deletion (MATLAB rules).

s = ["a" "b" "c"; "d" "e" "f"];
assert(isequal(s(2, :), ["d" "e" "f"]));
assert(isequal(size(s(:, [1 3])), [2 2]));
t = s([1 4]);
assert(isequal(size(t), [1 2]) && isequal(t, ["a" "e"]));
assert(isequal(size(s([1; 4])), [2 1]));
assert(isequal(size(s(:)), [6 1]));
t = s(logical([1 0 1; 0 1 0]));
assert(isequal(t, ["a"; "e"; "c"]));
assert(s(end) == "f" && s(end, 1) == "d" && s(2, 3) == "f");
assert(isequal(size(s([])), [0 0]) && isequal(size(s(1, [])), [1 0]));

% A string scalar is a 1x1 array
u = "abc";
assert(u(1) == "abc" && u(1, 1) == "abc");
v = u([1 1 1]);
assert(isequal(size(v), [1 3]) && all(v == "abc"));

% Errors
try
  t = s(7); %#ok<NASGU>
  error('expected an error');
catch e
  assert(strcmp(e.message, 'Index exceeds the number of array elements. Index must not exceed 6.'));
end
try
  t = s(3, 1); %#ok<NASGU>
  error('expected an error');
catch e
  assert(strcmp(e.message, 'Index in position 1 exceeds array bounds. Index must not exceed 2.'));
end

% Stores, growth and deletion
s(3, 1) = "g";
assert(isequal(size(s), [3 3]) && s(3, 1) == "g");
s(:, 2) = [];
assert(isequal(size(s), [3 2]) && isequal(s(1, :), ["a" "c"]));
w = ["x" "y" "z"];
w(2) = [];
assert(isequal(w, ["x" "z"]));
w(5) = "q";
assert(isequal(size(w), [1 5]) && w(5) == "q");
w(1:2) = ["m" "n"];
assert(isequal(w(1:2), ["m" "n"]));
w([1 2]) = "k";
assert(isequal(w(1:2), ["k" "k"]));
z = ["p"; "q"];
z(4) = "r";
assert(isequal(size(z), [4 1]));
z(2, 2) = "r";
assert(isequal(size(z), [4 2]));
c = ["a" "b" "c"];
c(2) = 5;
assert(c(2) == "5");
e = strings(0, 0);
e(2) = "b";
assert(isequal(size(e), [1 2]) && e(2) == "b");

% Brace indexing gives char vectors
y = s{2, 1};
assert(ischar(y) && strcmp(y, 'd'));
q = ["one" "two"];
q{2} = 'three';
assert(q(2) == "three" && isstring(q));

disp('SUCCESS');
