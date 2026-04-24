function bench_07_chained_elemwise_reduce()
    % Level 07 -- chained tensor_local feeding a reduction.
    %
    % Body:
    %   c = a .* b;         % tensor_local, elemwise
    %   d = sqrt(c + 1);    % tensor_local whose elemExpr references c
    %   s = s + sum(d) * i; % reduction on d, multiplied by loop var
    %
    % The whole-loop JIT must fuse through the chain: emit an inline
    % inner loop over the tensor length that computes
    % sqrt(a[j]*b[j] + 1) and accumulates into __sum_d, with NO
    % materialization of c or d. Scope check: d's elemExpr contains
    % a Var ref to c (another tensor_local) -- the inner emitter
    % substitutes c's elemExpr at the use site.

    n = 200000;
    k = 32;
    a = linspace(0.1, 1.0, k);
    b = linspace(-0.5, 0.5, k);

    % Warmup
    s = 0;
    for i = 1:1000
        c = a .* b;
        d = sqrt(c + 1);
        s = s + sum(d) * i;
    end

    s = 0;
    t0 = tic;
    for i = 1:n
        c = a .* b;
        d = sqrt(c + 1);
        s = s + sum(d) * i;
    end
    t_elapsed = toc(t0);

    % sum over i of (sum(sqrt(a.*b + 1)) * i)
    %   = sum(sqrt(a.*b + 1)) * sum(1:n)
    sd = sum(sqrt(a .* b + 1));
    expected = sd * n * (n + 1) / 2;
    assert(abs(s - expected) / abs(expected) < 1e-10, ...
        sprintf('level 07: s=%.6e expected=%.6e', s, expected));

    per_iter_ns = 1e9 * t_elapsed / n;
    fprintf('BENCH: test=chained_elemwise_reduce n=%d k=%d elapsed=%.6f per_iter_ns=%.2f s=%.6e\n', ...
        n, k, t_elapsed, per_iter_ns, s);
    disp('SUCCESS')
end
