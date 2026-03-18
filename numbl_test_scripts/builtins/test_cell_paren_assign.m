% Test that c(i) = {val} assigns val directly into cell slot (no double-wrapping)
c = cell(2, 1);
c(1) = {eye(3)};
assert(~iscell(c{1}));
assert(isequal(c{1}, eye(3)));

% c(i) = 1x1_cell should store the cell's content
tmp = {42};
c(2) = tmp;
assert(~iscell(c{2}));
assert(c{2} == 42);

% Multiple assignments
d = cell(1, 3);
d(1) = {[1 2 3]};
d(2) = {'hello'};
d(3) = {sparse(eye(2))};
assert(isequal(d{1}, [1 2 3]));
assert(strcmp(d{2}, 'hello'));
assert(issparse(d{3}));

disp('SUCCESS')
