% Test calling a local function from a nested function.
% Local functions are siblings of the main function (defined at script top-level
% after the main code), while nested functions live inside a parent function.
% A nested function should be able to invoke its parent's sibling local functions.

% Test 1: Nested function calls local helper directly
result1 = outer1(3);
assert(result1 == 9)  % nested calls square_local(3) = 9

% Test 2: Nested function calls local helper that returns multiple values
[a, b] = outer2(4);
assert(a == 4)
assert(b == 16)

% Test 3: Nested function uses local helper combined with parent workspace
result3 = outer3(5);
assert(result3 == 35)  % (x + offset) * 5 = (5 + 2) * 5 = 35, with offset captured from parent

% Test 4: Local function called from nested handle returned to caller
h = outer4(10);
v = h(3);
assert(v == 13)  % add_local(10, 3) = 13

disp('SUCCESS')

function result = outer1(x)
    function y = inner()
        y = square_local(x);
    end
    result = inner();
end

function [p, q] = outer2(x)
    function [a, b] = inner()
        [a, b] = pair_local(x);
    end
    [p, q] = inner();
end

function result = outer3(x)
    offset = 2;
    function r = inner()
        r = scale_local(x + offset, x);
    end
    result = inner();
end

function h = outer4(base)
    function r = inner(delta)
        r = add_local(base, delta);
    end
    h = @inner;
end

function y = square_local(x)
    y = x * x;
end

function [a, b] = pair_local(x)
    a = x;
    b = x * x;
end

function r = scale_local(value, factor)
    r = value * factor;
end

function r = add_local(a, b)
    r = a + b;
end
