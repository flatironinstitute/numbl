% struct() with a cell-valued field yields a 1x1 struct whose fields can
% be indexed directly (s.a(2), s.c{1}, s.f(x)).
s = struct('a', [1 2], 'c', {{'n'}}, 'f', @(x) x + 1);
assert(isequal(size(s), [1 1]));
assert(s.a(2) == 2, 's.a(2)');
assert(strcmp(s.c{1}, 'n'), 's.c{1}');
assert(s.f(3) == 4, 's.f(3)');
assert(s.a(end) == 2, 's.a(end)');
s.a(2) = 7;
assert(isequal(s.a, [1 7]));

t = struct('p', {[4 5 6]});
assert(t.p(end) == 6);
u = struct('q', {struct('r', [8 9])});
assert(u.q.r(2) == 9);

% A 1x1 struct array obtained by deleting elements behaves the same way
v = struct('a', {[1 2], [3 4]});
v(1) = [];
assert(numel(v) == 1);
assert(v.a(2) == 4, 'v.a(2)');

disp('SUCCESS');
