% Mirror the chunkie adapgausskerneval inner-loop pattern:
%
%   vals = zeros(1, maxdepth);
%   for ii = 1:ntarg
%       vals(:, 1) = oneintp(-1, 1, ...);   % col-assign with scalar RHS
%       ...
%       v2 = oneintp(...);                  % oneintp returns 1x1 → scalar
%       vals(:, jj+1) = v2;                 % col-assign with scalar Ident RHS
%   end
%
% Each line below was a separate JIT bail in the original code:
%
% (a) `vals(:, 1) = oneintp(...)` — col-LHS with non-Ident RHS that returns
%     a scalar (because the matmul collapses 1×1 to a number at runtime).
%     Required: tryLowerColAssign to accept non-Ident RHS, mMul JIT type
%     for known-1×1 result to predict `number` (matching unwrap1x1).
%
% (b) Workspace-file local function call (`oneintp` is defined later in
%     the same .m file). Required: resolveUserFunction to handle
%     `localFunction` with `source.from === "workspaceFile"`.
%
% (c) Probing with `oneintp(-1, 1, …)` — `-1` parses as Unary(Neg, 1) and
%     wasn't constant-folded, so probe rejected it as "non-Var/NumberLit".
%     Required: track `exact` through unary minus and use `literalNumber`
%     in the probe.
%
% (d) `vals(:, jj+1) = v2` where v2 is a scalar Ident. Required:
%     tryLowerColAssign Ident-branch to also accept scalar Ident when
%     the dst column has size 1.

function jit_loop_chunkie_main()
    n = 33;
    a = ones(1, n);
    b = ones(n, 1);
    vals = zeros(1, 200);

    for ii = 1:5
        %!numbl:assert_jit
        vals(:, 1) = oneintp_scalar(-1, 1, a, b);
        v2 = oneintp_scalar(-1, 0, a, b);
        vals(:, ii + 1) = v2;
    end

    expected_first = sum(a) * sum(b) / 1.0;  % sentinel pass-through: not used
    assert(vals(1, 1) == n, '1: scalar-RHS col-assign');
    assert(vals(1, 2) == n, '2: scalar-Ident-RHS col-assign');
    assert(expected_first > 0, 'sentinel');
end

function v = oneintp_scalar(lo, hi, a, b)
    % Mirror oneintp's structure: scale, matmul, return scalar (via
    % unwrap1x1 when the matmul result is 1x1).
    u = (hi - lo) / 2;
    v = a * b;       % 1×n times n×1 → scalar at runtime (unwrap1x1)
    v = v + u * 0;   % keep result type as number
end

jit_loop_chunkie_main();
disp('SUCCESS');
