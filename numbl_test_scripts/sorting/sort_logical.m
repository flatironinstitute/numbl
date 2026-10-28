% sort on logical inputs returns logical values (and double indices).

s = sort(true);
assert(islogical(s) && s == true);
[s, i] = sort(false);
assert(islogical(s) && s == false && i == 1 && isa(i, 'double'));

a = logical([1 0 1]);
[s, i] = sort(a);
assert(islogical(s));
assert(isequal(s, logical([0 1 1])));
assert(isequal(i, [2 1 3]) && isa(i, 'double'));

s = sort(a, 'descend');
assert(islogical(s) && isequal(s, logical([1 1 0])));

[s, i] = sort(logical([1 0; 0 1]), 2);
assert(islogical(s) && isequal(s, logical([0 1; 0 1])));
assert(isequal(i, [2 1; 1 2]));

s = sort(logical([1 0; 0 1; 1 1]), 1, 'descend');
assert(islogical(s) && isequal(s, logical([1 1; 1 1; 0 0])));

s = sort(logical([]));
assert(islogical(s) && isempty(s));

% Through a function called in a loop (JIT-eligible at --opt 1/2).
for k = 1:3
  [s, i] = sort_logical_helper(logical([1 0 1 0]));
end
assert(islogical(s) && isequal(s, logical([1 1 0 0])));
assert(isequal(i, [1 3 2 4]));

disp('SUCCESS')

function [s, i] = sort_logical_helper(a)
  [s, i] = sort(a, 'descend');
end
