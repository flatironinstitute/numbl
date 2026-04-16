function scalar_bench()
    % Scalar-hot-loop benchmark: numbl --opt 0/1/2 vs MATLAB vs Octave.
    % See benchmarks/scalar_bench.md for full description + typical results.
    %
    % Written as a function file (not a script) because Octave 9 doesn't
    % accept local functions at the end of a script file. MATLAB, Octave,
    % and numbl all invoke the main function automatically.

    N = 60000;    % outer points
    M = 500;      % inner series terms
    fprintf('N=%d, M=%d (total sin/div calls: %d)\n', N, M, N*M);
    fprintf('----------------------------------------\n');

    % Warm-up: one call lands the JIT specialization in cache so the
    % timed call below measures pure execution, not compile or load time.
    warm = run_bench(100, 10);
    fprintf('warmup check = %.6f\n', warm);

    t = tic;
    total = run_bench(N, M);
    elapsed = toc(t);

    fprintf('result  = %.12f\n', total);
    fprintf('elapsed = %.4f s\n', elapsed);
    fprintf('throughput = %.2f Mcalls/s\n', (N*M) / elapsed / 1e6);

    disp('SUCCESS')
end

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
