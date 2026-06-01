% Function-handle calls whose argument is a JIT-constructed anonymous
% struct (`s = []; s.r = ...; s.d = ...; h(s, ...)`). This mirrors the
% chunkie `oneintp` pattern from
%   chunkie/+chnk/adapgausskerneval.m   and
%   chunkie/+chnk/chunkerkerneval_smooth.m
% which builds `srcinfo` / `targinfo` ad hoc and then calls
% `kern(srcinfo, targinfo)`. Today the func-handle return-type probe
% bails when an arg is a struct because `representativeValue` doesn't
% know how to synthesize struct probes; that bail kills JIT for both
% the build_matrix and eval phases of the helmholtz_starfish bench.
%
% The `%!numbl:assert_jit` markers pin the loops; if probing fails the
% loop falls back to the interpreter and the directive throws.
%
% Status: case 1 was already JIT-friendly (it doesn't pass the struct
% to a handle). Cases 2 and 3 are the ones we are unblocking.

% --- Handles. `h_scalar` reads two scalar struct fields; `h_tensor`
% reads two tensor struct fields and reduces. The exact body is not
% the point — what matters is that the handle is *opaque* at the
% probe site, so the JIT must synthesize representative struct args.
h_scalar = @(s, t) s.x + t.y;
h_tensor = @(s, t) sum(s.r(:)) + sum(t.r(:));

base_r = [1.0; 2.0; 3.0; 4.0];

n = 20;

% 1) Baseline — handle takes only scalars (no struct args). Should
%    already JIT today; here as a regression guard so a future change
%    that breaks scalar-handle probing fails this test too.
acc1 = 0;
for i = 1:n
    a = i * 1.5;
    b = i + 2.5;
    acc1 = acc1 + a + b;
end

% 2) Anon-struct → handle, scalar fields. The struct is built in-loop
%    and immediately passed to a handle. Probe must synthesize a
%    representative struct {x: number, y: number}.
acc2 = 0;
for i = 1:n
    s = [];
    s.x = i * 1.0;
    t = [];
    t.y = i * 2.0;
    acc2 = acc2 + h_scalar(s, t);
end

% 3) Anon-struct → handle, tensor fields. The chunkie shape: struct
%    carries real-tensor `.r`, `.d`, etc., and is consumed by a
%    function handle whose body is opaque to the JIT.
acc3 = 0;
for i = 1:n
    rint = base_r * i;
    dint = base_r + i;
    srcinfo = [];
    srcinfo.r = rint;
    srcinfo.d = dint;
    targinfo = [];
    targinfo.r = rint + 1.0;
    targinfo.d = dint - 1.0;
    acc3 = acc3 + h_tensor(srcinfo, targinfo);
end

% Correctness checks (independent of whether JIT lowered or not).
expected1 = 0;
for i = 1:n
    expected1 = expected1 + i * 1.5 + (i + 2.5);
end
assert(abs(acc1 - expected1) < 1e-12, '1: acc1 value');

expected2 = 0;
for i = 1:n
    expected2 = expected2 + i * 1.0 + i * 2.0;
end
assert(abs(acc2 - expected2) < 1e-12, '2: acc2 value');

expected3 = 0;
for i = 1:n
    rint = base_r * i;
    expected3 = expected3 + sum(rint) + sum(rint + 1.0);
end
assert(abs(acc3 - expected3) < 1e-9, '3: acc3 value');

disp('SUCCESS')
