% struct([]) creates an empty struct array; struct arrays support s(:).

% struct([]) -> empty struct
s = struct([]);
assert(isstruct(s));
assert(isempty(s));
assert(numel(s) == 0);

% struct-array colon indexing
t = struct('type', {1, 2, 3}, 'ref', {10, 20, 30});
t2 = t(:);
assert(numel(t2) == 3);
assert(t2(2).ref == 20);

% The [arr(:).field] comma-list expansion idiom
vals = [t(:).type];
assert(isequal(vals, [1 2 3]));

% Colon indexing a scalar struct yields the struct itself
u.a = 5;
u2 = u(:);
assert(u2.a == 5);
assert(isequal([u(:).a], 5));

disp('SUCCESS');
