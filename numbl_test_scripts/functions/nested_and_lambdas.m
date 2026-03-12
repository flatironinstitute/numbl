% Test combining nested functions, lambdas, and function handles
% in tricky configurations.

% Test 1: Lambda inside nested function captures by value (not reference)
result1 = lambda_in_nested(10);
assert(result1 == 10)  % lambda captured k=10 before k was changed to 99

% Test 2: Nested function returns lambda that closes over shared state
[inc, get] = make_state(0);
inc(3);
inc(7);
v2 = get();
assert(v2 == 10)

% Test 3: Multiple nested functions coordinating shared state
result3 = state_machine();
assert(result3 == 42)

% Test 4: Nested function returning another nested function's handle
[push, pop, peek] = make_stack();
push(10);
push(20);
push(30);
assert(peek() == 30)
assert(pop() == 30)
assert(pop() == 20)
assert(peek() == 10)

% Test 5: Lambda capturing variable that nested function later modifies
result5 = lambda_vs_nested();
assert(result5 == 100)  % lambda captured snapshot, nested func changed original

disp('SUCCESS')

function result = lambda_in_nested(x)
    function r = inner()
        k = 10;
        f = @(a) a + k;  % lambda captures k=10 by value
        k = 99;           % change k after lambda creation
        r = f(0);         % should return 10, not 99
    end
    result = inner();
end

function [inc, get] = make_state(init)
    val = init;
    function do_inc(n)
        val = val + n;
    end
    function r = do_get()
        r = val;
    end
    inc = @do_inc;
    get = @do_get;
end

function result = state_machine()
    state = 0;
    function reset()
        state = 0;
    end
    function add(n)
        state = state + n;
    end
    function multiply(n)
        state = state * n;
    end
    function r = read()
        r = state;
    end
    add(10);       % state = 10
    multiply(3);   % state = 30
    add(5);        % state = 35
    reset();       % state = 0
    add(6);        % state = 6
    multiply(7);   % state = 42
    result = read();
end

function [push, pop, peek] = make_stack()
    data = zeros(1, 100);
    n = 0;
    function do_push(v)
        n = n + 1;
        data(n) = v;
    end
    function r = do_pop()
        r = data(n);
        n = n - 1;
    end
    function r = do_peek()
        r = data(n);
    end
    push = @do_push;
    pop = @do_pop;
    peek = @do_peek;
end

function result = lambda_vs_nested()
    x = 100;
    snap = @() x;       % lambda captures x=100 by value
    function change()
        x = 999;         % nested func changes x by reference
    end
    change();
    result = snap();     % should return 100 (captured value), not 999
end
