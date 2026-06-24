% Accessing a property on an object array yields a comma-separated list (so
% [arr.prop] concatenates the per-element values), and `end` inside an index of
% an object array resolves to the array length.

arr = [ArrItem_(10), ArrItem_(20), ArrItem_(30), ArrItem_(40), ArrItem_(50)];

% Property access across the array -> comma-separated list.
assert(isequal([arr.v], [10 20 30 40 50]));

% `end` resolves to the last element.
last = arr(end);
assert(last.v == 50);

% `end` inside a range expression.
sub = arr(end-2:end);
assert(numel(sub) == 3);
assert(isequal([sub.v], [30 40 50]));

disp('SUCCESS')
