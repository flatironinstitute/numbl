% Stage 24 soft-bail UserCall correctness test.
%
% When a JIT-friendly loop body calls a user function whose body
% can't be lowered, the JIT now emits a UserDispatchCall that
% dispatches through the interpreter at runtime rather than bailing
% the whole loop.
%
% The probe-at-compile-time path must:
%   - Return correct numeric results across iterations
%   - Return correct tensor results + propagate through downstream reads
%   - Not fire for callees using evalin/assignin/dbstack (caller-aware
%     builtins that can't be probed safely)

addpath(fileparts(mfilename('fullpath')));

% 1) Scalar-returning soft-bail
total1 = 0;
for i = 1:50
    total1 = total1 + soft_bail_scalar(i);
end
assert(total1 == 2 * (50 * 51 / 2), '1: scalar soft-bail sum');

% 2) Tensor-returning soft-bail + downstream scalar reads
total2 = 0;
for i = 1:50
    v = soft_bail_tensor(i);
    total2 = total2 + v(1) + v(2) + v(3);
end
% v = [i, 2i, 3i]; sum = 6i; total = 6 * 50*51/2 = 7650
assert(total2 == 6 * (50 * 51 / 2), '2: tensor soft-bail sum');

% 3) A caller-aware callee must NOT soft-bail — it either still
%    lowers normally or stays in the interpreter. The test is just
%    that the result is correct (evalin sees the caller's workspace).
outer_q = 42;
got = caller_reads_outer_q();
assert(got == 42, '3: evalin callee resolves caller correctly');

disp('SUCCESS');

function out = soft_bail_scalar(x)
    % bsxfun with function-handle arg isn't JIT-lowerable (current)
    out = sum(bsxfun(@plus, [x; x], 0));
end

function v = soft_bail_tensor(x)
    v = bsxfun(@times, [1; 2; 3], x);
end

function v = caller_reads_outer_q()
    v = evalin('caller', 'outer_q', -1);
end
