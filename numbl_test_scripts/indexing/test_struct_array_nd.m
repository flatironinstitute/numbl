% N-d struct arrays: creation, growth, deletion, reads and shape builtins
% (MATLAB rules).

% Creation by indexed assignment pads with empty fields
clear s
s(2, 3).a = 1;
assert(isequal(size(s), [2 3]) && numel(s) == 6);
assert(isempty(s(1, 1).a) && isa(s(1, 1).a, 'double') && isequal(size(s(1, 1).a), [0 0]));
assert(s(2, 3).a == 1 && s(6).a == 1);
x = [];
x(3).f = 2;
assert(isequal(size(x), [1 3]) && isempty(x(1).f) && x(3).f == 2);
w = struct('a', 1);
w(2, 1).a = 3;
assert(isequal(size(w), [2 1]) && w(2).a == 3);

% struct() takes the size of its cell arguments
t = struct('a', {});
assert(isequal(size(t), [0 0]) && isempty(t) && isequal(fieldnames(t), {'a'}));
u = struct('a', {1 2; 3 4});
assert(isequal(size(u), [2 2]));
assert(isequal(size(struct('a', {1; 2; 3})), [3 1]));
assert(isequal(size(struct([])), [0 0]));

% Comma-separated lists expand in column-major order
assert(isequal([u.a], [1 3 2 4]));
c = {u.a};
assert(isequal(size(c), [1 4]) && isequal(c, {1, 3, 2, 4}));

% Reads: linear, subscripts, colon, end, logical masks, ranges
assert(u(2, 1).a == 3 && u(3).a == 2 && u(end).a == 4 && u(end, 1).a == 3);
v = u(:);
assert(isequal(size(v), [4 1]) && isequal([v.a], [1 3 2 4]));
r = u(2, :);
assert(isequal(size(r), [1 2]) && isequal([r.a], [3 4]));
r = u(:, 1);
assert(isequal(size(r), [2 1]) && isequal([r.a], [1 3]));
r = u([1 4]);
assert(isequal(size(r), [1 2]) && isequal([r.a], [1 4]));
r = u([1; 4]);
assert(isequal(size(r), [2 1]));
r = u(logical([1 0; 0 1]));
assert(isequal(size(r), [2 1]) && isequal([r.a], [1 4]));
s7 = struct('a', {1 2 3});
r = s7([true false true]);
assert(isequal(size(r), [1 2]) && isequal([r.a], [1 3]));
s8 = struct('a', {1; 2; 3});
assert(isequal(size(s8([1 3])), [2 1]));
assert(isequal(size(s7([])), [0 0]) && isequal(size(s7(:, [])), [1 0]));
assert(isequal(size(s7(1, :)), [1 3]));

% Growth keeps vector orientation; out-of-range reads error
s5 = struct('a', {1 2 3});
s5(5).a = 7;
assert(isequal(size(s5), [1 5]) && isempty(s5(4).a));
s6 = struct('a', {1; 2; 3});
s6(5).a = 7;
assert(isequal(size(s6), [5 1]));
s9 = struct('a', {1 2; 3 4});
s9(:, 3) = struct('a', 9);
assert(isequal(size(s9), [2 3]) && isequal([s9.a], [1 3 2 4 9 9]));
try
  y = s7(4); %#ok<NASGU>
  error('expected an error');
catch e
  assert(~isempty(strfind(e.message, 'Index exceeds')));
end
try
  s9(3, 4, 2) = []; %#ok<NASGU>
  error('expected an error');
catch e
  assert(~strcmp(e.message, 'expected an error'));
end

% Assignment of structs and struct arrays
s = struct('a', {1 2 3 4});
s(2:3) = struct('a', {7 8});
assert(isequal([s.a], [1 7 8 4]));
s([1 4]) = struct('a', 0);
assert(isequal([s.a], [0 7 8 0]));
s = struct('a', 1, 'b', 2);
s(2) = struct('b', 3, 'a', 4);
assert(isequal(fieldnames(s), {'a'; 'b'}) && isequal([s.a], [1 4]));
s = struct('a', {1 2 3});
try
  s(2) = struct('b', 1);
  error('expected an error');
catch e
  assert(strcmp(e.message, 'Subscripted assignment between dissimilar structures.'));
end
s(2).b = 'x';
assert(isequal(fieldnames(s), {'a'; 'b'}) && isempty(s(3).b) && s(2).b == 'x');

% Deletion
u = struct('a', {1 2; 3 4});
u(2) = [];
assert(isequal(size(u), [1 3]) && isequal([u.a], [1 2 4]));
u2 = struct('a', {1 2; 3 4});
u2(:, 1) = [];
assert(isequal(size(u2), [2 1]) && isequal([u2.a], [2 4]));
s = struct('a', 1);
s(1) = [];
assert(isequal(size(s), [1 0]) && isstruct(s) && isequal(fieldnames(s), {'a'}));
t3 = struct('a', {1 2; 3 4; 5 6});
t3(:, :, 2) = t3;
assert(isequal(size(t3), [3 2 2]) && ndims(t3) == 3 && size(t3, 3) == 2);
v = t3(:, 2, 2);
assert(isequal(size(v), [3 1]) && isequal([v.a], [2 4 6]));
t3(:, :, 1) = [];
assert(isequal(size(t3), [3 2]));
try
  t3(1, 1) = [];
  error('expected an error');
catch e
  assert(strcmp(e.message, 'A null assignment can have only one non-colon index.'));
end

% Shape builtins
r = reshape(struct('a', {1 2 3 4 5 6}), 2, 3);
assert(isequal(size(r), [2 3]) && r(2, 2).a == 4);
p = permute(r, [2 1]);
assert(isequal(size(p), [3 2]) && isequal([p.a], [1 3 5 2 4 6]));
tr = r';
assert(isequal(size(tr), [3 2]) && isequal([tr.a], [1 3 5 2 4 6]));
q = repmat(struct('a', 1), 2, 3);
assert(isequal(size(q), [2 3]));
assert(isequal(size(cat(3, r, r)), [2 3 2]));
assert(isequal(size([r; r]), [4 3]) && isequal(size([r, r]), [2 6]));
assert(isequal(size([struct('a', {}), struct('a', 1)]), [1 1]));
assert(numel(r) == 6 && length(r) == 3 && ndims(r) == 2);
assert(isrow(struct('a', {1 2})) && iscolumn(struct('a', {1; 2})) && ~isvector(r));

% arrayfun keeps the shape of a struct array
m = arrayfun(@(e) e.a * 2, struct('a', {1 2; 3 4}));
assert(isequal(m, [2 4; 6 8]));
m = arrayfun(@(e) e.a, r, 'UniformOutput', false);
assert(isequal(size(m), [2 3]) && m{2, 2} == 4);

% for iterates over columns
n = 0;
for e = struct('a', {1 2 3})
  n = n + e.a;
end
assert(n == 6);
k = 0;
for e = struct('a', {1 2; 3 4})
  k = k + 1;
  assert(isequal(size(e), [2 1]));
end
assert(k == 2);

% struct2cell / cell2struct / num2cell / isequal
c2 = struct2cell(struct('a', {1 2 3}, 'b', {4 5 6}));
assert(isequal(size(c2), [2 1 3]) && c2{2, 1, 3} == 6);
assert(isequal(size(cell2struct({1 2; 3 4}, {'a', 'b'}, 1)), [2 1]));
assert(isequal(size(cell2struct({1 2; 3 4}, {'a', 'b'}, 2)), [2 1]));
nc = num2cell(r);
assert(isequal(size(nc), [2 3]) && nc{2, 2}.a == 4);
assert(isequal(struct('a', {1 2}), struct('a', {1 2})));
assert(~isequal(struct('a', {1 2}), struct('a', {1; 2})));
assert(~isequal(struct('a', {1 2}), struct('a', {1 3})));

disp('SUCCESS');
