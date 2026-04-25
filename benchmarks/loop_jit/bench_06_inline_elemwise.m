function bench_06_inline_elemwise()
    % Level 06 -- inline elementwise + reduction inside the loop.
    %
    % Each iteration evaluates a Green's-function-style kernel
    %   c = b ./ sqrt(a.*a + (i*dt)^2)
    % and reduces it with sum. This is the shape of a 1/r-style
    % potential evaluation where `a` holds in-panel offsets, `b` holds
    % charges, and `i*dt` is the target's perpendicular distance --
    % i.e. the inner kernel that chunkie-adjacent code actually runs.
    %
    % Because the sqrt argument depends on both the per-element value
    % `a` and the per-iteration scalar `i*dt`, the inner reduction is
    % NOT loop-invariant and NOT reassociable into (loop-invariant sum)
    % + (per-iter scalar). That keeps the inner reduction honestly
    % running n*k work even after GCC's LICM + reassociation passes --
    % matching what MATLAB is forced to do.
    %
    % The whole-loop JIT should still fuse this into a single inline
    % inner C loop that computes `b[j] / sqrt(a[j]*a[j] + ti*ti)` and
    % accumulates into __sum_c with no `c` tensor materialized.

    n = 200000;
    k = 32;
    a = linspace(0.1, 1.0, k);
    b = linspace(-0.5, 0.5, k);
    dt = 0.001;

    % Warmup
    s = 0;
    for i = 1:1000
        c = b ./ sqrt(a .* a + (i * dt) * (i * dt));
        s = s + sum(c);
    end

    s = 0;
    t0 = tic;
    for i = 1:n
        c = b ./ sqrt(a .* a + (i * dt) * (i * dt));
        s = s + sum(c);
    end
    t_elapsed = toc(t0);

    % Reference: sum_i sum_j b[j] / sqrt(a[j]^2 + (i*dt)^2), vectorised.
    i_col = (1:n)' * dt;
    ker = b ./ sqrt(a .* a + i_col .* i_col);    % n x k via broadcasting
    expected = sum(ker(:));
    rel_err = abs(s - expected) / abs(expected);
    assert(rel_err < 1e-10, ...
        sprintf('level 06: s=%.10e expected=%.10e rel_err=%.3e', ...
            s, expected, rel_err));

    per_iter_ns = 1e9 * t_elapsed / n;
    fprintf('BENCH: test=inline_elemwise n=%d k=%d elapsed=%.6f per_iter_ns=%.2f s=%.6e\n', ...
        n, k, t_elapsed, per_iter_ns, s);
    disp('SUCCESS')
end
