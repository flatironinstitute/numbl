% [d1, ..., dk] = size(A) in compiled code: the last output is the product
% of the remaining dimensions, as in the interpreter and MATLAB.

x3 = zeros(2, 3, 4);
[a, b] = sz2(x3);
assert(a == 2 && b == 12);
[a, b, c] = sz3(x3);
assert(isequal([a b c], [2 3 4]));
[a, b, c, d] = sz4(x3);
assert(isequal([a b c d], [2 3 4 1]));
[a, b] = sz2(zeros(5, 6));
assert(a == 5 && b == 6);
[a, b, c] = sz3(zeros(5, 6));
assert(isequal([a b c], [5 6 1]));

disp('SUCCESS')

function [a, b] = sz2(x)
    %!numbl:assert_jit c
    [a, b] = size(x);
end

function [a, b, c] = sz3(x)
    %!numbl:assert_jit c
    [a, b, c] = size(x);
end

function [a, b, c, d] = sz4(x)
    %!numbl:assert_jit c
    [a, b, c, d] = size(x);
end
