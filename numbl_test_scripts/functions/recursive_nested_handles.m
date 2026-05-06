% Recursive and mutually-recursive nested functions reached through
% a handle that escapes the parent. The handle's closure references
% itself (or a sibling handle) by the same name it has in the parent
% function — MATLAB semantics: nested functions share parent workspace
% by reference, so the closure can call the handle by its parent-local
% name.

% Test 1: Recursive nested fn called via output handle.
g = make_recursive();
y = g(5);
assert(y == 15)  % 5+4+3+2+1+0

% Test 2: Mutual recursion via two output handles.
[hEven, hOdd] = make_parity();
assert(hEven(0) == true)
assert(hOdd(0) == false)
assert(hEven(4) == true)
assert(hOdd(3) == true)
assert(hEven(7) == false)

% Test 3: Closure references the output handle by its name and chains.
g2 = make_chain();
y2 = g2(5);
assert(y2 == 5)  % adds 1 five times

disp('SUCCESS')

function h = make_recursive()
    function r = f(n)
        if n <= 0
            r = 0;
            return
        end
        r = n + h(n - 1);
    end
    h = @f;
end

function [hEven, hOdd] = make_parity()
    function r = isEven(n)
        if n == 0
            r = true;
            return
        end
        r = hOdd(n - 1);
    end
    function r = isOdd(n)
        if n == 0
            r = false;
            return
        end
        r = hEven(n - 1);
    end
    hEven = @isEven;
    hOdd = @isOdd;
end

function h = make_chain()
    function r = step(n)
        if n <= 0
            r = 0;
            return
        end
        r = h(n - 1) + 1;
    end
    h = @step;
end
