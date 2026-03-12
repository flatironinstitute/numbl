% Test that [a; b; c] with class instances calls vertcat, NOT horzcat.
% This verifies that semicolons in array construction don't spuriously
% dispatch to horzcat on individual elements.

a = VertOnly_(1);
b = VertOnly_(2);
c = VertOnly_(3);

% [a; b; c] should call vertcat(a, b, c), never horzcat
result = [a; b; c];
assert(isequal(result.data, [1; 2; 3]), 'vertcat should combine data');

% Single element in brackets should not call horzcat either
d = [a];
assert(isequal(d.data, 1), 'single element should pass through');

disp('SUCCESS')
