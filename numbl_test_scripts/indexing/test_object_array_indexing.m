% Object arrays: reads, stores, growth and deletion (MATLAB rules).

p = [IdxObj_(1) IdxObj_(2) IdxObj_(3) IdxObj_(4)];
q = p([1 3]);
assert(isequal(size(q), [1 2]) && isequal([q.x], [1 3]));
q = p(:);
assert(isequal(size(q), [4 1]));
q = p(logical([1 0 1 1]));
assert(isequal([q.x], [1 3 4]));
assert(p(end).x == 4);
q = p([]);
assert(isequal(size(q), [0 0]) && isa(q, 'IdxObj_'));

r = reshape(p, 2, 2);
q = r(2, :);
assert(isequal([q.x], [2 4]));
q = r(:, 1);
assert(isequal(size(q), [2 1]) && isequal([q.x], [1 2]));
assert(isequal(size(r([1 4])), [1 2]));
assert(r(end, end).x == 4 && r(3).x == 3);

% A scalar object indexed with all-ones subscripts is itself
o = IdxObj_(7);
assert(o(1).x == 7 && o(1, 1).x == 7 && o(end).x == 7);
oo = o([1 1 1]);
assert(isequal(size(oo), [1 3]) && isequal([oo.x], [7 7 7]));

% Errors
try
  q = p(5); %#ok<NASGU>
  error('expected an error');
catch e
  assert(strcmp(e.message, 'Index exceeds the number of array elements. Index must not exceed 4.'));
end
try
  q = p(0); %#ok<NASGU>
  error('expected an error');
catch e
  assert(strcmp(e.message, 'Array indices must be positive integers or logical values.'));
end
try
  q = r(3, 1); %#ok<NASGU>
  error('expected an error');
catch e
  assert(strcmp(e.message, 'Index in position 1 exceeds array bounds. Index must not exceed 2.'));
end

% Deletion
p(2) = [];
assert(isequal([p.x], [1 3 4]) && isequal(size(p), [1 3]));
r(:, 1) = [];
assert(isequal(size(r), [2 1]) && isequal([r.x], [3 4]));
d = IdxObj_(1);
d(1) = [];
assert(isequal(size(d), [1 0]) && isa(d, 'IdxObj_'));

% Stores and growth keep orientation
c = p';
assert(isequal(size(c), [3 1]));
c(5) = IdxObj_(9);
assert(isequal(size(c), [5 1]) && c(5).x == 9);
c(2:3) = [IdxObj_(7) IdxObj_(8)];
assert(c(2).x == 7 && c(3).x == 8);
c([1 5]) = IdxObj_(0);
assert(c(1).x == 0 && c(5).x == 0);
d = IdxObj_(1);
d(2, 2) = IdxObj_(5);
assert(isequal(size(d), [2 2]) && d(2, 2).x == 5 && d(1, 1).x == 1);
d(1, :) = [];
assert(isequal(size(d), [1 2]) && d(2).x == 5);

% Element property updates
p = [IdxObj_(1) IdxObj_(2) IdxObj_(3)];
for k = 1:3
  p(k).x = 10 * k;
end
assert(isequal([p.x], [10 20 30]));

% for iterates over columns; arrayfun keeps the shape
n = 0;
for e = p
  n = n + e.x;
end
assert(n == 60);
m = arrayfun(@(e) e.x, reshape([p IdxObj_(40)], 2, 2));
assert(isequal(m, [10 30; 20 40]));

disp('SUCCESS');
