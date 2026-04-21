function complex_tensor_bench()
    % Complex-tensor hot-loop benchmark: numbl --opt 0/1/2 vs MATLAB.
    % Companion to complex_scalar_bench.m and tensor_ops_bench.m. See
    % benchmarks/complex_tensor_bench.md for full description.
    %
    % Six kernels exercise the paired-buffer complex codegen across
    % fusion-eligible and fusion-ineligible shapes:
    %
    %   1. Mandelbrot step  (z.*z + c)          — fused
    %   2. Two-tensor chain (z.*w + y)          — fused
    %   3. Conj chain       (conj(z).*z + y)    — fused
    %   4. Real→complex widen (x + y*1i)        — fused
    %   5. Complex divide   (z./w)              — per-op (Smith's method)
    %   6. abs + sum        (acc += sum(abs(z))) — per-op (abs bails fusion)

    N = 500000;      % element count
    trials = 100;    % inner repeats per kernel

    fprintf('N=%d, trials=%d\n', N, trials);
    fprintf('----------------------------------------\n');

    % Deterministic inputs (no RNG) so numbl / MATLAB agree bit-for-bit.
    x = build_range(N) * 1e-7;
    y = build_range(N) * 2e-7;
    z = x + y * 1i;

    % Warm-up lands the JIT specializations before we time.
    [~, ~, ~, ~, ~, ~, ~, ~, ~] = run_all(x, y, z, 2);

    % Hot: real measurement.
    [t1, t2, t3, t4, t5, t6, u1, u2, acc] = run_all(x, y, z, trials);

    total = t1 + t2 + t3 + t4 + t5 + t6;

    fprintf('1. Mandelbrot z.*z+c:       %7.3f s   (fused)\n', t1);
    fprintf('2. Tensor chain z.*w+y:     %7.3f s   (fused)\n', t2);
    fprintf('3. Conj chain conj(z).*z:   %7.3f s   (fused)\n', t3);
    fprintf('4. Widening x+y*1i:         %7.3f s   (fused)\n', t4);
    fprintf('5. Divide z./w:             %7.3f s   (per-op)\n', t5);
    fprintf('6. abs + sum reduction:     %7.3f s   (per-op)\n', t6);
    fprintf('----------------------------------------\n');
    fprintf('elapsed = %.3f s\n', total);

    fprintf('\nCheck values (must match across runtimes):\n');
    fprintf('  real(sum(u1))  = %.10g\n', real(sum(u1)));
    fprintf('  imag(sum(u1))  = %.10g\n', imag(sum(u1)));
    fprintf('  real(sum(u2))  = %.10g\n', real(sum(u2)));
    fprintf('  imag(sum(u2))  = %.10g\n', imag(sum(u2)));
    fprintf('  abs_acc        = %.10g\n', acc);
    disp('SUCCESS')
end

function r = build_range(N)
    % Deterministic [1, 2, ..., N] row vector. Avoids `1:N` which the
    % C-JIT Range path would bail on when called from a hot script.
    r = zeros(1, N);
    for i = 1:N
        r(i) = i;
    end
end

% Runs all six kernels `trials` times each and returns per-kernel
% wall-clocks plus the last written u1 / u2 tensors and the abs-sum
% accumulator, so the outer can print check values that tie any
% correctness regression back to a specific kernel.
function [t1, t2, t3, t4, t5, t6, u1, u2, acc] = run_all(x, y, z, trials)
    u1 = z;
    u2 = z;

    c = 0.001 + 0.001i;
    w = z + c;  % complex tensor, used by kernels 2 / 5

    % ── 1. Mandelbrot step: z .* z + c (scalar broadcast) ──────────────
    tic;
    for k = 1:trials
        u1 = z .* z + c;
    end
    t1 = toc;

    % ── 2. Two-tensor chain: z .* w + y ────────────────────────────────
    % Complex × complex tensor, then complex + real tensor.
    tic;
    for k = 1:trials
        u1 = z .* w + y;
    end
    t2 = toc;

    % ── 3. Conj chain: conj(z) .* z + y ────────────────────────────────
    % Equivalent to |z|^2 + y (imag cancels by construction).
    tic;
    for k = 1:trials
        u1 = conj(z) .* z + y;
    end
    t3 = toc;

    % ── 4. Real→complex widening: x + y * 1i ───────────────────────────
    % Build a complex tensor from two real ones. ImagLiteral fuses.
    tic;
    for k = 1:trials
        u1 = x + y * 1i;
    end
    t4 = toc;

    % ── 5. Complex divide: z ./ w ──────────────────────────────────────
    % Fusion rejects complex `./` (Smith's method branches don't SIMD);
    % this kernel runs per-op against numbl_complex_binary_elemwise.
    tic;
    for k = 1:trials
        u2 = z ./ w;
    end
    t5 = toc;

    % ── 6. abs + sum reduction: acc += sum(abs(z)) ─────────────────────
    % abs(complex) is a type-transition call (complex → real) that
    % fusion doesn't yet handle, and the reduction itself runs after.
    tic;
    acc = 0;
    for k = 1:trials
        acc = acc + sum(abs(z));
    end
    t6 = toc;
end
