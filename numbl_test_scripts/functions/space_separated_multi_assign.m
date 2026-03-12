% Test space-separated multi-output assignment

AA = ones(3, 5);
[m n] = size(AA);
assert(m == 3);
assert(n == 5);

% Mixed with tilde
[x, ~, z] = three_outputs();
assert(x == 1);
assert(z == 3);

disp('SUCCESS');

function [a, b, c] = three_outputs()
    a = 1;
    b = 2;
    c = 3;
end
