% Regression: inside [ ... ] / { ... }, whitespace before '{' separates
% elements (a following cell literal) rather than indexing the preceding
% expression. [c {x}] is horzcat(c, {x}); [c{x}] indexes c. numbl already
% handled the '(' case ([a (b)]); this covers the '{' case, which was
% mis-parsed as an index — pulseq's spoilBlockContents=[spoilBlockContents
% {mr.makeLabel(...)}] hit it with "Cannot convert struct to number".

c = {10, 20};

% Space before '{' => concatenation (new cell element)
x = [c {30}];
assert(numel(x) == 3);
assert(x{3} == 30);

% Works with a struct element too (the original failing case)
L.type = 'labelset';
L.label = 'LIN';
L.value = 1;
y = [c {L}];
assert(numel(y) == 3);
assert(y{3}.value == 1);

% No space => still indexing (no regression)
assert([c{2}] == 20);
assert([c{end}] == 20);

% Cell literals with space-separated elements
z = {10 20 30};
assert(numel(z) == 3);

% Mixed index-then-literal inside a cell literal
f = {7};
w = {f{1} 8};
assert(w{1} == 7 && w{2} == 8);

% The '(' analogue keeps working
a = 1; b = 2;
m = [a (b)];
assert(numel(m) == 2 && m(2) == 2);

disp('SUCCESS');
