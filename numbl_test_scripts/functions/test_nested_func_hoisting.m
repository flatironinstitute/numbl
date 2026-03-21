% Test that nested functions are visible before their definition (hoisting).
% In MATLAB, nested functions can be called before their textual definition.

result = outer(5);
assert(result == 25, 'nested function hoisting failed');

result2 = outer_multi(3, 4);
assert(result2 == 7, 'nested function hoisting with multiple nested funcs failed');

disp('SUCCESS');

function y = outer(x)
    % Call nested function that is defined AFTER this call site
    y = square(x);

    function r = square(v)
        r = v * v;
    end
end

function y = outer_multi(a, b)
    % Call both nested functions before their definitions
    y = add(a, b);

    function r = add(x, y)
        r = x + y;
    end

    function r = mul(x, y)
        r = x * y;
    end
end
