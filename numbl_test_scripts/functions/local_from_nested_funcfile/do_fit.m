function h = do_fit(a, b)
    function y = inner(z)
        % Nested function calls TWO sibling local helpers of the parent.
        y = combine_local(a, b, z) + add_local(a, b);
    end
    h = @inner;
end

function r = combine_local(a, b, z)
    r = a * z + b;
end

function r = add_local(a, b)
    r = a + b;
end
