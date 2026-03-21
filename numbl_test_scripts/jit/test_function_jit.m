% Test function-level JIT: scalar functions compiled to JS

function y = cube(x)
    y = x * x * x;
end

function y = sigmoid(x)
    y = 1 / (1 + exp(-x));
end

% Call functions in a loop to exercise both function and loop JIT
s = 0;
for i = 1:100
    s = s + cube(i);
end
assert(s == 25502500, 'cube sum wrong');

% Sigmoid should produce values in (0, 1)
v = sigmoid(0);
assert(abs(v - 0.5) < 1e-10, 'sigmoid(0) should be 0.5');
v = sigmoid(10);
assert(v > 0.99, 'sigmoid(10) should be near 1');
v = sigmoid(-10);
assert(v < 0.01, 'sigmoid(-10) should be near 0');

disp('SUCCESS')
