% arrayfun over a struct array passes each scalar struct element to the fn
s(1).coefs = [1 2 NaN];
s(2).coefs = [3 4 5];
s(3).coefs = [6 7 8];

% Field access followed by indexing inside the lambda
first = arrayfun(@(x) x.coefs(1), s);
assert(isequal(first, [1 3 6]));

% any/isnan over a field, with (:) linearization
hasnan = arrayfun(@(x) any(isnan(x.coefs(:))), s);
assert(islogical(hasnan));
assert(isequal(hasnan, logical([1 0 0])));

% UniformOutput=false collects per-element results into a cell
c = arrayfun(@(x) x.coefs * 2, s, 'UniformOutput', false);
assert(iscell(c));
assert(numel(c) == 3);
assert(isequal(c{2}, [6 8 10]));

disp('SUCCESS')
