% Test that anonymous functions propagate nargout to the functions they call.
% In MATLAB, [a, b] = f(x) where f = @(x) multi_out(x) should return 2 outputs.

% Basic case: anonymous function wrapping a multi-output function
f = @(x) multi_out(x);
[a, b] = f(5);
assert(a == 10, 'Expected a=10');
assert(b == 25, 'Expected b=25');

% With extra captured variable (closure)
k = 3;
g = @(x) multi_out_with_extra(x, k);
[a, b] = g(4);
assert(a == 7, 'Expected a=7');
assert(b == 12, 'Expected b=12');

% Single output should still work
c = f(3);
assert(c == 6, 'Expected c=6');

disp('SUCCESS');

function [doubled, squared] = multi_out(x)
    doubled = x * 2;
    squared = x^2;
end

function [sum_val, prod_val] = multi_out_with_extra(x, k)
    sum_val = x + k;
    prod_val = x * k;
end
