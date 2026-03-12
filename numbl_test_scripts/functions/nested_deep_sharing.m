% Test deep nesting with shared state, lambdas, and returned handles.
% Exercises multi-level reference sharing, lambda value capture,
% and nested-function-returned handles.

[get, snap, mutate] = aaa();

% After aaa(): x was set to 2 at the end
assert(get() == 2)      % nested func sees x=2 by reference
assert(snap() == 1)     % lambda captured x=1 at creation time

% mutate() calls c which does x=x+1 and returns handle to d
dfun = mutate();
assert(get() == 3)      % x is now 3

% dfun() calls d which does x=x+10
dfun();
assert(get() == 13)     % x is now 13

% snap still returns the captured value
assert(snap() == 1)

% Mutate again
dfun2 = mutate();       % x=x+1 -> x=14
assert(get() == 14)
dfun2();                % x=x+10 -> x=24
assert(get() == 24)

% Original dfun still works on the same shared x
dfun();                 % x=x+10 -> x=34
assert(get() == 34)

disp('SUCCESS')

function [get, snap, mutate] = aaa()
    x = 1;
    function r = b()
        r = x;           % return x by reference
    end
    function dfun = c()
        x = x + 1;       % mutate shared x
        function d()
            x = x + 10;  % deeply nested, also mutates shared x
        end
        dfun = @d;
    end
    get = @b;
    snap = @() x;         % lambda captures x=1 at creation time
    mutate = @c;
    x = 2;
end
