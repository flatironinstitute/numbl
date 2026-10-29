% The logical class survives the operations that copy, reshape, grow or
% shrink a logical array, and concatenation follows MATLAB's class rules.

% Copy-on-write indexed assignment
a = logical([1 0 1]);
b = a;
b(2) = true;
assert(strcmp(class(a), 'logical') && isequal(a, logical([1 0 1])));
assert(strcmp(class(b), 'logical') && isequal(b, true(1, 3)));
c = a;
c(1) = 0;
assert(strcmp(class(c), 'logical') && isequal(c, logical([0 0 1])));
d = a;
d(3) = 5;
assert(strcmp(class(d), 'logical') && isequal(d, logical([1 0 1])));
m = true(2, 2);
n = m;
n(:, 1) = false;
assert(strcmp(class(n), 'logical') && isequal(n, logical([0 1; 0 1])));
assert(strcmp(class(m), 'logical') && all(m(:)));

% Deletion
e = logical([1 0 1 1]);
e(2) = [];
assert(strcmp(class(e), 'logical') && isequal(e, true(1, 3)));
f = true(2, 3);
f(:, 2) = [];
assert(strcmp(class(f), 'logical') && isequal(size(f), [2 2]));
g = logical([1 0 1]);
g([]) = [];
assert(strcmp(class(g), 'logical'));

% Growth
h = true(1, 2);
h(5) = true;
assert(strcmp(class(h), 'logical') && isequal(h, logical([1 1 0 0 1])));
k = false(2);
k(3, 3) = true;
assert(strcmp(class(k), 'logical') && isequal(size(k), [3 3]) && nnz(k) == 1);
p = logical([]);
p(3) = true;
assert(strcmp(class(p), 'logical') && isequal(p, logical([0 0 1])));

% Concatenation: all logical stays logical, a double operand makes double
assert(strcmp(class([true false]), 'logical'));
assert(strcmp(class([true; false]), 'logical'));
assert(strcmp(class([logical([1 0]); logical([0 1])]), 'logical'));
assert(strcmp(class([true 1]), 'double'));
assert(strcmp(class([true(1, 2), 2]), 'double'));
assert(strcmp(class([true(2, 1), [3; 4]]), 'double'));
assert(strcmp(class(cat(1, true(2, 1), false)), 'logical'));
assert(strcmp(class(cat(3, true(2), false(2))), 'logical'));
assert(strcmp(class(cat(2, true, 2)), 'double'));
assert(strcmp(class(horzcat(true, false)), 'logical'));
assert(strcmp(class(vertcat(true, 0)), 'double'));
assert(strcmp(class([true(1, 2), []]), 'double'));
x = [];
x = [x, true];
assert(strcmp(class(x), 'double') && isequal(x, 1));
assert(strcmp(class([[]; true(2, 1)]), 'double'));
assert(strcmp(class([logical([]), true]), 'logical'));

% Reshape, permute, transpose, repmat
q = logical([1 0 1; 0 1 0]);
assert(strcmp(class(reshape(q, 3, 2)), 'logical'));
assert(strcmp(class(reshape(q, [], 1)), 'logical'));
assert(strcmp(class(permute(q, [2 1])), 'logical'));
assert(strcmp(class(permute(true(2, 3, 4), [3 1 2])), 'logical'));
assert(strcmp(class(q'), 'logical'));
assert(strcmp(class(q.'), 'logical'));
assert(strcmp(class(repmat(q, 2, 3)), 'logical'));
assert(strcmp(class(repmat(true, 2, 2)), 'logical'));
assert(strcmp(class(squeeze(true(1, 1, 3))), 'logical'));
assert(strcmp(class(q(:)), 'logical'));
assert(strcmp(class(q(2, :)), 'logical'));
assert(strcmp(class(q(1)), 'logical'));

% Sort and unique
assert(strcmp(class(sort(q)), 'logical'));
[s, si] = sort(q, 2, 'descend');
assert(strcmp(class(s), 'logical') && strcmp(class(si), 'double'));
assert(strcmp(class(sort(logical([1 0 1]))), 'logical'));
assert(strcmp(class(unique(q)), 'logical'));
[u, ui] = unique(q, 'rows');
assert(strcmp(class(u), 'logical') && strcmp(class(ui), 'double'));

% Arithmetic on logicals gives double
assert(strcmp(class(q + q), 'double'));
assert(strcmp(class(-q), 'double'));
assert(strcmp(class(q * 2), 'double'));

disp('SUCCESS')
