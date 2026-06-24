% A function nested inside another nested function must be callable from its
% parent, regardless of definition order, and shares the whole scope chain
% (variables of all enclosing functions). Mirrors the rktoolbox rat_krylov
% pattern, where a nested `run_krylov` calls a sibling-nested `continuation_pair`.

result = outer();
assert(isequal(result, [42, 7]), 'doubly-nested sharing/order failed');

disp('SUCCESS');

function out = outer()
    a = 0;          % owned by outer
    b = 0;          % owned by outer
    mid();          % nested fn that itself contains a nested fn
    out = [a, b];

    function mid()
        % Call a sibling nested fn defined LATER in this same body.
        deep();
        function deep()
            a = 42;             % writes outer's a (shared across both levels)
            b = inc(b);         % calls a top-level helper
        end
    end
end

function y = inc(x)
    y = x + 7;
end
