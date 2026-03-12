% Test that anonymous functions capture variable VALUES at creation time
% (MATLAB semantics: capture by value, not by reference)

% Basic: changing variable after lambda creation should NOT affect the lambda
k = 10;
add_k = @(x) x + k;
k = 20;
assert(add_k(5) == 15)  % Should be 15 (captured k=10), not 25

% Multiple captured variables
a = 1;
b = 2;
f = @(x) x + a + b;
a = 100;
b = 200;
assert(f(0) == 3)  % Should be 3 (captured a=1, b=2), not 300

% Capture in a loop - classic closure test
funcs = cell(1, 3);
for i = 1:3
    funcs{i} = @(x) x + i;
end
% Each lambda should have captured the value of i at creation time
assert(funcs{1}(0) == 1)
assert(funcs{2}(0) == 2)
assert(funcs{3}(0) == 3)

disp('SUCCESS')
