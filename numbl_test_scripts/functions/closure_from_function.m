% Test returning lambda functions from functions
% The lambda should capture values from the function's local scope

% Basic: function returns a lambda that captures a local variable
f = make_adder(10);
assert(f(5) == 15)
assert(f(0) == 10)

% The returned lambda should be independent of subsequent calls
g = make_adder(100);
assert(g(5) == 105)
assert(f(5) == 15)  % f should still use its captured value of 10

% Function that returns a lambda capturing multiple locals
h = make_linear(3, 7);
assert(h(0) == 7)
assert(h(1) == 10)
assert(h(5) == 22)

% Function that returns a lambda capturing a computed value
counter = make_counter(5);
assert(counter() == 6)

disp('SUCCESS')

function f = make_adder(n)
    f = @(x) x + n;
end

function f = make_linear(slope, intercept)
    f = @(x) slope * x + intercept;
end

function f = make_counter(start)
    val = start + 1;
    f = @() val;
end
