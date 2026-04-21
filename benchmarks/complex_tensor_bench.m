function complex_tensor_bench()
    % Complex-tensor hot-loop benchmark: numbl --opt 0/1/2 vs MATLAB.
    % Companion to complex_scalar_bench.m. See
    % benchmarks/complex_tensor_bench.md for full description.
    %
    % Inner step: `z = z .* z + c` on a length-N complex tensor, repeated
    % M times. Exercises the paired (re, im) buffer codegen:
    %   .*          → numbl_complex_binary_elemwise (tensor × tensor)
    %   + c         → numbl_complex_scalar_binary_elemwise (tensor + scalar)
    % Result check is `real(sum(z))`, which exercises complex reduction
    % (numbl_complex_flat_reduce) plus `real()` on a complex scalar.

    N = 200000;   % tensor length
    M = 500;      % inner iterations
    fprintf('N=%d, M=%d (total z.*z+c tensor ops: %d)\n', N, M, N * M);
    fprintf('----------------------------------------\n');

    x = build_seed(N);

    % Warm-up lands the JIT specializations before we time.
    warm = run_bench(x, 5);
    fprintf('warmup check = %.6f\n', warm);

    t = tic;
    total = run_bench(x, M);
    elapsed = toc(t);

    fprintf('result     = %.12f\n', total);
    fprintf('elapsed    = %.4f s\n', elapsed);
    fprintf('throughput = %.2f Mops/s\n', (N * M) / elapsed / 1e6);

    disp('SUCCESS')
end

function x = build_seed(N)
    x = zeros(1, N);
    for i = 1:N
        x(i) = i * 1e-8;
    end
end

% Runs the Mandelbrot-style map `z = z.*z + c` for M iterations across
% a full length-N complex tensor, then reduces via real(sum(z)). Pulls
% seeding out so the timed loop is pure complex-tensor arithmetic.
function total = run_bench(x, M)
    c = 0.001 + 0.001i;
    z = x + x * 2i;             % complex tensor [N]
    for k = 1:M
        z = z .* z + c;
    end
    s = sum(z);                 % complex scalar
    total = real(s);
end
