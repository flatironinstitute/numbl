% MATLAB name-resolution priority for `pkg.fn(args)`-style calls:
% the first segment must be NOT-a-local-variable for the namespace
% interpretation to win. If a local with the same name exists, the
% expression is field-then-call on that value.
%
% These tests pin the priority because the JIT lowering can be tempted
% to short-circuit any `MethodCall(Ident, name, args)` to a namespace
% call. Whenever that happens, the result must match what MATLAB
% actually does — i.e. the variable wins.
%
% Each test runs both code paths (JIT, if it triggers, and interpreted
% — by virtue of the warm-up iterations and any bailouts) and asserts
% the result matches the interpreter-only ground truth.

% --- Test 1: local struct named like the namespace, used inside a loop.
% mymath.add_two as a struct field handle multiplies. namespace's
% mymath.add_two adds. If JIT incorrectly takes the namespace path,
% we get the sum instead of the product → mismatch.
mymath = struct('add_two', @(a, b) a * b);
total_local = 0;
n = 50;
for i = 1:n
    total_local = total_local + mymath.add_two(i, i + 1);
end
expected_local = 0;
for i = 1:n
    expected_local = expected_local + i * (i + 1);
end
assert(total_local == expected_local, ...
    '1: local mymath struct must shadow +mymath namespace');

% --- Test 2: namespace with no local var. Same loop shape — adds.
clear mymath
total_ns = 0;
for i = 1:n
    total_ns = total_ns + mymath.add_two(i, i + 1);
end
expected_ns = 0;
for i = 1:n
    expected_ns = expected_ns + (i + (i + 1));
end
assert(total_ns == expected_ns, '2: namespace mymath.add_two adds');

% --- Test 3: variable-then-namespace alternation across calls.
% First clear, then set, then clear. The JIT shouldn't cache one
% interpretation across this transition — bailout/dispatch must
% follow the runtime env.
total_alt = 0;
for j = 1:5
    if mod(j, 2) == 0
        mymath = struct('add_two', @(a, b) a * b);
    end
    total_alt = total_alt + mymath.add_two(j, j + 1);
    if mod(j, 2) == 0
        clear mymath
    end
end
% odd j (1,3,5): namespace adds → j + (j+1)
% even j (2,4):  local struct multiplies → j*(j+1)
expected_alt = (1 + 2) + 2 * 3 + (3 + 4) + 4 * 5 + (5 + 6);
assert(total_alt == expected_alt, ...
    '3: alternating local/namespace must follow runtime env');

% --- Test 4: function parameter named like the namespace shadows it.
function out = call_via_param(mymath, n)
    out = 0;
    for i = 1:n
        out = out + mymath.add_two(i, i + 1);
    end
end

s = struct('add_two', @(a, b) a * b);
total_param = call_via_param(s, n);
expected_param = expected_local;
assert(total_param == expected_param, ...
    '4: function param named mymath must shadow +mymath namespace');

% --- Test 5: scalar-typed local with the namespace name should error
% on field access (no field 'add_two' on a number) — error must come
% out, not a silent namespace dispatch.
mymath = 42;
threw = false;
try
    x = mymath.add_two(1, 2);
catch
    threw = true;
end
assert(threw, '5: mymath as scalar must error, not silently call namespace');

disp('SUCCESS')
