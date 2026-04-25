function bench_07_chained_elemwise_reduce()
    % Level 07 -- chained tensor_local feeding a reduction.
    %
    % Body:
    %   c = a .* b;               % tensor_local, elemwise, loop-invariant
    %   d = sqrt(c + 1 + i*1e-6); % tensor_local whose elemExpr references c
    %                             %   AND depends on the loop variable, so
    %                             %   sum(d) cannot be hoisted out of the
    %                             %   outer loop (the +1 keeps the argument
    %                             %   strictly positive given a.*b can be
    %                             %   slightly negative)
    %   s = s + sum(d) * i;       % reduction on d, multiplied by loop var
    %
    % The whole-loop JIT must fuse through the chain: emit an inline
    % inner loop over the tensor length that computes
    % sqrt(a[j]*b[j] + i*0.001) and accumulates into __sum_d, with NO
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
        d = sqrt(c + 1 + i * 1e-6);
        s = s + sum(d) * i;
    end

    s = 0;
    t0 = tic;
    for i = 1:n
        c = a .* b;
        d = sqrt(c + 1 + i * 1e-6);
        s = s + sum(d) * i;
    end
    t_elapsed = toc(t0);

    % Reference: sum_i i * sum_j sqrt(a[j]*b[j] + 1 + i*1e-6), vectorised.
    ab = a .* b;
    i_col = (1:n)';
    d_mat = sqrt(ab + 1 + i_col * 1e-6);    % n x k via broadcasting
    per_iter = sum(d_mat, 2);             % n x 1
    expected = sum(i_col .* per_iter);
    rel_err = abs(s - expected) / abs(expected);
    assert(rel_err < 1e-10, ...
        sprintf('level 07: s=%.10e expected=%.10e rel_err=%.3e', ...
            s, expected, rel_err));

    per_iter_ns = 1e9 * t_elapsed / n;
    fprintf('BENCH: test=chained_elemwise_reduce n=%d k=%d elapsed=%.6f per_iter_ns=%.2f s=%.6e\n', ...
        n, k, t_elapsed, per_iter_ns, s);
    disp('SUCCESS')
end
