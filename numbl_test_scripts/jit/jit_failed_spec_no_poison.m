% One function the JIT cannot emit must not stop later ones from compiling.
%
% Every emitted module contains every spec in the shared `Lowerer`, so a
% spec that lowers cleanly but throws during *emit* — a builtin that rejects
% in `emitJs`, here a broadcast comparison on a runtime-shaped tensor —
% used to stay in the map and make every subsequent `compileSpec` in the
% session rethrow its error. One unsupported line in one helper would then
% quietly stop unrelated functions from being JIT'd for the rest of the run.
% `compileSpec` now rolls back whatever it added when it fails.

X = rand(4, 4) + 1;
Y = rand(4, 4) + 1;

% Attempt (and fail) to compile a function with an unsupported construct.
u = cannot_emit(X, Y);
assert(u > 0, 'the interpreter still runs the declined function');

% An unrelated function compiled afterwards must still JIT.
assert(fine(30) == 216225, 'later spec still compiles');

disp('SUCCESS')

function u = cannot_emit(X, Y)
th = atan2(Y, X);
th(th < 1e-13) = th(th < 1e-13) + 2*pi;   % tensor-vs-scalar compare: declines
u = sum(th(:));
end

function s = fine(n)
%!numbl:assert_jit
s = 0;
for i = 1:n
    for j = 1:n
        s = s + i * j;
    end
end
end
