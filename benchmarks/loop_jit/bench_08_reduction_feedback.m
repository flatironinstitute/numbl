function bench_08_reduction_feedback()
    % Level 08 -- scalar reduction feeds back into a later tensor_local.
    %
    % Body:
    %   r = sqrt(a + i*0.01);   % tensor_local (depends on loop var
    %                           %   per-element, so sum(r) can't be hoisted)
    %   q = sum(r);             % reduction -> scalar q
    %   t = sqrt(b + q);        % tensor_local depending on the reduction
    %                           %   scalar; again per-element non-linear so
    %                           %   sum(t) can't be factored through b
    %   s = s + sum(t);         % final reduction into inout scalar
    %
    % Tests: two separate reduction loops in the same iteration, with
    % the first reduction's scalar result feeding an expression inside
    % the second tensor_local. No materialization of r or t; each
    % reduction runs its own inlined inner loop. The sqrt wrappers are
    % what prevent GCC's LICM + reassociation passes from rewriting
    % each inner sum into (loop-invariant reduction of a/b) plus a
    % per-iter scalar.

    n = 200000;
    k = 32;
    a = linspace(0.1, 1.0, k);
    b = linspace(-0.5, 0.5, k);

    % Warmup
    s = 0;
    for i = 1:1000
        r = sqrt(a + i * 0.01);
        q = sum(r);
        t = sqrt(b + q);
        s = s + sum(t);
    end

    s = 0;
    t0 = tic;
    for i = 1:n
        r = sqrt(a + i * 0.01);
        q = sum(r);
        t = sqrt(b + q);
        s = s + sum(t);
    end
    t_elapsed = toc(t0);

    % Reference: replicate the exact iteration order so rounding matches.
    expected = 0;
    for i = 1:n
        ri = sqrt(a + i * 0.01);
        qi = sum(ri);
        ti = sqrt(b + qi);
        expected = expected + sum(ti);
    end
    rel_err = abs(s - expected) / max(1, abs(expected));
    assert(rel_err < 1e-10, ...
        sprintf('level 08: s=%.10e expected=%.10e rel_err=%.3e', ...
            s, expected, rel_err));

    per_iter_ns = 1e9 * t_elapsed / n;
    fprintf('BENCH: test=reduction_feedback n=%d k=%d elapsed=%.6f per_iter_ns=%.2f s=%.6e\n', ...
        n, k, t_elapsed, per_iter_ns, s);
    disp('SUCCESS')
end
