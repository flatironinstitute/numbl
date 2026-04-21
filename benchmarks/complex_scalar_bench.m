function complex_scalar_bench()
    % Complex-scalar hot-loop benchmark: numbl --opt 0/1/2 vs MATLAB.
    % See benchmarks/complex_scalar_bench.md for full description.
    %
    % The inner function is a Mandelbrot-style squared-plus-offset
    % iteration `z = z*z + c` which exercises the C-JIT's pair-of-
    % doubles complex scalar codegen (complex mul + complex add per
    % iteration, two loads, two stores).

    N = 50000;    % outer points
    M = 400;      % inner iterations per point
    fprintf('N=%d, M=%d (total z*z+c ops: %d)\n', N, M, N*M);
    fprintf('----------------------------------------\n');

    % Warm-up lands the JIT specialization before we time.
    warm = run_bench(100, 10);
    fprintf('warmup check = %.6f\n', warm);

    t = tic;
    total = run_bench(N, M);
    elapsed = toc(t);

    fprintf('result  = %.12f\n', total);
    fprintf('elapsed = %.4f s\n', elapsed);
    fprintf('throughput = %.2f Mops/s\n', (N*M) / elapsed / 1e6);

    disp('SUCCESS')
end

% Sums real(z_final) across N starting points, each iterated M times
% through the map z -> z^2 + c with c = 0.001 + 0.001i. Every inner
% step is complex mul + complex add — no branches, no real scalars.
function total = run_bench(N, M)
    total = 0.0;
    c = 0.001 + 0.001i;
    for i = 1:N
        z = (i * 1e-6) + (i * 2e-6) * 1i;
        for k = 1:M
            z = z * z + c;
        end
        total = total + real(z);
    end
end
