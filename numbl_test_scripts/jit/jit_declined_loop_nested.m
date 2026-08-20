% A loop the JIT declines must not drag everything under it down with it.
%
% The loop and call executors gate on `loopDepth > 0`, on the assumption
% that an enclosing loop will be compiled and will swallow whatever is
% nested inside. When that loop is *declined* the assumption is false, and
% before `jitDeclinedLoopDepth` existed the whole subtree — including
% expensive callees with their own hot loops — ran interpreted. On the
% L-shaped conjugate-gradient solve in the tutorials that was a factor of
% about 30.
%
% Here the outer loop is declined (it indexes a cell), and the assert
% inside the callee pins that the callee is compiled anyway.

names = {'a', 'b'};
acc = 0;
for k = 1:3
    acc = acc + sweep(20) + numel(names{1});
end
assert(acc == 3 * (2100 + 1), 'nested call under a declined loop');

% Same for a nested loop rather than a nested call.
acc2 = 0;
for k = 1:3
    inner = 0;
    for j = 1:20
        %!numbl:assert_jit
        inner = inner + j * j;
    end
    acc2 = acc2 + inner + numel(names{2});
end
assert(acc2 == 3 * (2870 + 1), 'nested loop under a declined loop');

disp('SUCCESS')

function s = sweep(n)
%!numbl:assert_jit
s = 0;
for i = 1:n
    for j = 1:n
        s = s + i - j / 2;
    end
end
end
