% Capture-free function handles that are LOOP INPUTS (defined before the
% loop, called inside it) are inlined as in-scope `@...` constants in the
% JIT spec, so the loop C-JITs instead of declining on the opaque handle.
% The `%!numbl:assert_jit c` markers fail the test if a loop falls back to
% the interpreter at --opt 2, so we know the JIT path is exercised.

% --- Case 1: anonymous, capture-free handle as a loop input.
fa = @(t) t.^2 + 1;
acc1 = 0;
for i = 1:1000
    %!numbl:assert_jit c
    acc1 = acc1 + fa(i);
end

% --- Case 2: named-function handle as a loop input.
fn = @sq;
acc2 = 0;
for i = 1:1000
    %!numbl:assert_jit c
    acc2 = acc2 + fn(i);
end

% --- Case 3: a capture-FUL handle stays a runtime input (not inlinable),
% so the loop declines to the interpreter — but the result must still be
% correct. No assert_jit here (it is expected to NOT C-JIT).
k = 7;
fc = @(t) t + k;
acc3 = 0;
for i = 1:1000
    acc3 = acc3 + fc(i);
end

% --- Correctness (independent of whether the JIT or interpreter ran).
expected1 = sum((1:1000).^2 + 1);
assert(abs(acc1 - expected1) < 1e-6, '1: anon loop-input handle');

expected2 = sum((1:1000).^2);
assert(abs(acc2 - expected2) < 1e-6, '2: named loop-input handle');

expected3 = sum((1:1000) + 7);
assert(abs(acc3 - expected3) < 1e-6, '3: capture-ful handle correctness');

disp('SUCCESS')

function y = sq(x)
    y = x.^2;
end
