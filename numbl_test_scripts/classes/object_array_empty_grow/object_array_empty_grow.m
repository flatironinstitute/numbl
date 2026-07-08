% Test ClassName.empty, growing an object array with end+1, and reshape

c = EmptyGrow_.empty(1, 0);
assert(numel(c) == 0);
assert(isempty(c));

c(end+1) = EmptyGrow_(10);   % grow the empty array
c(end+1) = EmptyGrow_(20);   % grow again (end on a single object)
c(end+1) = EmptyGrow_(30);
assert(numel(c) == 3);
assert(c(1).v == 10);
assert(c(2).v == 20);
assert(c(3).v == 30);

% reshape with an auto dimension
r = reshape(c, 1, []);
assert(isequal(size(r), [1 3]));
assert(r(2).v == 20);

col = reshape(c, [], 1);
assert(isequal(size(col), [3 1]));
assert(col(3).v == 30);

% no-argument form is 0x0
e = EmptyGrow_.empty;
assert(numel(e) == 0);

disp('SUCCESS')
