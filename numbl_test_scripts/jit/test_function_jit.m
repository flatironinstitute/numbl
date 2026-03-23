% Test function-level JIT: scalar functions compiled to JS

%!jit
function y = cube(x)
    y = x * x * x;
end

%!jit
function y = sigmoid(x)
    y = 1 / (1 + exp(-x));
end

a = cube(2);
assert(a == 8);

b = sigmoid(0);
assert(b == 1/2);

disp('SUCCESS');
