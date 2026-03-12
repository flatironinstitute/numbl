% Cell arrays

c = {1, 'hello', [1,2,3]};

% Access with {}
assert(c{1} == 1)
assert(strcmp(c{2}, 'hello'))
assert(length(c{3}) == 3)
assert(c{3}(2) == 2)

% iscell
assert(iscell(c))
assert(~iscell(42))

% numel of cell
assert(numel(c) == 3)

% Assign into cell
c{4} = 99;
assert(c{4} == 99)

% Cell with mixed types
d = {'a', 'b', 'c'};
assert(strcmp(d{2}, 'b'))

disp('SUCCESS')
