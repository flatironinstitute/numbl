% Scalar-hot-loop benchmark for comparing numbl's three optimization
% levels (--opt 0, 1, 2) against MATLAB.
%
% The inner function `run_bench` is a pure scalar loop over sin/arithmetic,
% which is exactly what the C-JIT (--opt 2) specializes best. No tensor,
% complex, or struct ops — so every call in the hot path stays on the
% C-JIT fast lane.
%
% Run in numbl (each opt level):
%   npx tsx src/cli.ts run examples/scalar_bench.m --opt 0
%   npx tsx src/cli.ts run examples/scalar_bench.m --opt 1
%   npx tsx src/cli.ts run examples/scalar_bench.m --opt 2
%
% Run in MATLAB:
%   matlab -batch "run('examples/scalar_bench.m')"
%
% One-shot cross-runtime comparison:
%   bash examples/scalar_bench_compare.sh
%
% All runs print the same 'result' value (to within floating-point
% rounding) alongside their wall time.
%
% Typical results (30M sin/div calls, Debian/gcc-14, x86-64):
%   --opt 0 (interpreter): ~31 s     (~1  Mcalls/s,  baseline)
%   --opt 1 (JS-JIT):      ~0.31 s   (~98 Mcalls/s,  ~100x over interp)
%   --opt 2 (C-JIT):       ~0.23 s   (~128 Mcalls/s, ~130x over interp)
%   MATLAB -batch:         ~0.37 s   (~81 Mcalls/s)
%
% Compile time (~50-60ms for cc + ~1ms for createRequire the .node) is
% amortized via a content-addressed disk cache at ~/.cache/numbl/c-jit/
% and is kept out of the timed section by the two warm-ups above.

N = 60000;    % outer points
M = 500;      % inner series terms
fprintf('N=%d, M=%d (total sin/div calls: %d)\n', N, M, N*M);
fprintf('----------------------------------------\n');

% Warm-up: one call lands the JIT specialization in cache so the timed
% call below measures pure execution, not compile or module-load time.
warm = run_bench(100, 10);
fprintf('warmup check = %.6f\n', warm);

t = tic;
total = run_bench(N, M);
elapsed = toc(t);

fprintf('result  = %.12f\n', total);
fprintf('elapsed = %.4f s\n', elapsed);
fprintf('throughput = %.2f Mcalls/s\n', (N*M) / elapsed / 1e6);

disp('SUCCESS')

% ── Hot function ──────────────────────────────────────────────────────
% Evaluates sum_{k=1..M} sin(x*k) / k^2 at N equally-spaced x-values and
% returns the grand sum. Pure scalar arithmetic, so every stmt/expr is
% in the C-JIT whitelist.
function total = run_bench(N, M)
    total = 0.0;
    for i = 1:N
        x = i * 0.001;
        acc = 0.0;
        for k = 1:M
            acc = acc + sin(x * k) / (k * k);
        end
        total = total + acc;
    end
end
