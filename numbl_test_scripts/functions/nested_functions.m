% Test nested functions (functions defined within functions)
% In MATLAB, nested functions share the parent workspace by reference,
% NOT by value capture like anonymous functions.

% Test 1: Basic nested function sees modified parent variable
result = outer(10);
assert(result == 20)  % inner sees x after x = x * 2

% Test 2: Nested function modifies parent variable
result2 = modifier_test(5);
assert(result2 == 15)  % inner adds 10 to x, parent returns modified x

% Test 3: Return nested function as handle and call it
h = make_counter(0);
v1 = h(1);  % increment by 1
v2 = h(1);  % increment by 1 again
v3 = h(5);  % increment by 5
assert(v1 == 1)
assert(v2 == 2)
assert(v3 == 7)

% Test 4: Multiple nested functions sharing parent workspace
result4 = multi_nested(3);
assert(result4 == 12)  % set_val sets x=10, add_val adds 2, result=12

% Test 5: Multi-level nesting (3 levels deep)
result5 = level1(3);
assert(result5 == 10)  % level3 returns x + y = 3 + 7 = 10

disp('SUCCESS')

function result = outer(x)
    function y = inner()
        y = x;  % should see x = 20 (by reference), not x = 10
    end
    x = x * 2;  % modify x after inner is defined
    result = inner();
end

function result = modifier_test(x)
    function do_modify()
        x = x + 10;  % modifies parent's x
    end
    do_modify();
    result = x;  % should be 15
end

function h = make_counter(start)
    count = start;
    function result = increment(n)
        count = count + n;
        result = count;
    end
    h = @increment;
end

function result = multi_nested(x)
    function set_val(v)
        x = v;  % modifies parent's x
    end
    function r = add_val(v)
        x = x + v;  % reads and modifies parent's x
        r = x;
    end
    set_val(10);  % x becomes 10
    result = add_val(2);  % x becomes 12, returns 12
end

function result = level1(x)
    function r = level2()
        function s = level3()
            s = x + y;  % sees both x and y from ancestors
        end
        y = 7;
        r = level3();  % returns x + y = 3 + 7 = 10
    end
    result = level2();
end
