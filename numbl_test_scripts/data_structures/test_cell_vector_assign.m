% Test cell array paren-indexed assignment with vector indices.
% In MATLAB, c(vector) = cellRHS assigns elements from cellRHS into
% the corresponding positions in c.

c = cell(1, 5);
c{1} = 'first';

% Assign multiple elements using a vector index
vals = {'second', 'third'};
c([2 3]) = vals;

assert(strcmp(c{1}, 'first'), 'c{1} should be first');
assert(strcmp(c{2}, 'second'), 'c{2} should be second');
assert(strcmp(c{3}, 'third'), 'c{3} should be third');

% Assign with a computed vector index (like l+(1:n))
l = 3;
numSubs = 2;
c(l+(1:numSubs)) = {'fourth', 'fifth'};

assert(strcmp(c{4}, 'fourth'), 'c{4} should be fourth');
assert(strcmp(c{5}, 'fifth'), 'c{5} should be fifth');

disp('SUCCESS');
