function chunkie_helmholtz_build()
    % Benchmark: isolated build step of the Helmholtz starfish CFIE.
    %
    % Sister benchmark to chunkie_helmholtz_starfish.m. That benchmark runs
    % the whole pipeline and reports one `build_matrix` phase. This one
    % targets only the build and breaks it into sub-phases so we can see
    % where the time actually goes:
    %
    %   besselh_only   - two besselh() calls on the N*N target-vs-source r
    %   kernel_only    - one fkern.eval(srcinfo,targinfo) over all points
    %                    (besselh + complex elemwise in helm2d.green/kern)
    %   smooth_matrix  - chnk.quadnative.buildmat over all (i,j) chunk pairs
    %   chunkermat     - full chunkermat incl. near/self GGQ corrections
    %   add_identity   - 0.5*eye(N) + sysmat
    %
    % Problem size matches the sister benchmark's hot run: narms=5, amp=0.5,
    % kvec=20*[1;-1.5] (zk~=36.06), producing 172 chunks with k=16 and a
    % 2752x2752 complex system matrix.
    %
    % See benchmarks/chunkie_helmholtz_build.md for results.

    t0 = tic;
    mip load --install chunkie;
    fprintf('BENCH: phase=chunkie_load t=%.6f\n', toc(t0));

    narms = 5;
    amp = 0.5;

    % ---- Warmup: small problem, same code paths ----
    t_warm = tic;
    run_build(narms, amp, 5*[1;-1.5]);
    fprintf('BENCH: phase=warmup t=%.6f\n', toc(t_warm));

    % ---- Hot measurement ----
    % kernel_only and smooth_matrix evaluate the kernel with identical src
    % and targ, so besselh hits r=0 on the diagonal; their fro norms come
    % out NaN and are dropped here. They're used for timing only.
    t_exec = tic;
    [chnkr_k, chnkr_nch, N, besselh_sum, ~, ~, sysmat_fro] = ...
        run_build(narms, amp, 20*[1;-1.5]);
    fprintf('BENCH: phase=execution t=%.6f\n', toc(t_exec));

    fprintf('CHECK: name=chnkr_k value=%.16e\n', double(chnkr_k));
    fprintf('CHECK: name=chnkr_nch value=%.16e\n', double(chnkr_nch));
    fprintf('CHECK: name=N value=%.16e\n', double(N));
    fprintf('CHECK: name=besselh_sum value=%.16e\n', besselh_sum);
    fprintf('CHECK: name=sysmat_fro value=%.16e\n', sysmat_fro);

    fprintf('DONE\n');
    disp('SUCCESS')
end

function [chnkr_k, chnkr_nch, N, besselh_sum, kernel_fro, sysmat_smooth_fro, sysmat_fro] = ...
         run_build(narms, amp, kvec)
    zk = norm(kvec);

    % discretize (not part of the build-step measurement)
    chnkr = chunkerfunc(@(t) starfish(t,narms,amp), struct('maxchunklen',4/zk));

    fkern = kernel('helm','c',zk,[1,-zk*1i]);
    kern_eval = fkern.eval;

    chnkr_k = chnkr.k;
    chnkr_nch = chnkr.nch;
    N = chnkr_k * chnkr_nch;
    opdims = [1; 1];

    % Flatten src/targ positions and normals once for the kernel-only and
    % besselh-only phases; matches what chnk.quadnative.buildmat does
    % internally for the full (1:nch, 1:nch) call.
    rstor = chnkr.rstor;
    nstor = chnkr.nstor;
    [dim, kk, ~] = size(rstor);
    rs_flat = reshape(rstor, dim, kk*chnkr_nch);
    ns_flat = reshape(nstor, dim, kk*chnkr_nch);

    % ---- Phase: besselh_only ----
    % Cost of the two Hankel calls in chnk.helm2d.green on the full N*N
    % distance matrix, nothing else.
    xs = repmat(rs_flat(1,:), N, 1);
    ys = repmat(rs_flat(2,:), N, 1);
    xt = repmat(rs_flat(1,:).', 1, N);
    yt = repmat(rs_flat(2,:).', 1, N);
    rx = xt - xs;
    ry = yt - ys;
    r_dist = sqrt(rx.*rx + ry.*ry);
    % Avoid besselh at r=0 (diagonal) by nudging; we only care about timing.
    r_dist(1:N+1:end) = 1;
    t0 = tic;
    h0 = besselh(0, 1, zk*r_dist);
    h1 = besselh(1, 1, zk*r_dist);
    fprintf('BENCH: phase=besselh_only t=%.6f\n', toc(t0));
    besselh_sum = abs(sum(h0(:))) + abs(sum(h1(:)));

    % ---- Phase: kernel_only ----
    % One fkern.eval call on the flattened src/targ. Includes besselh plus
    % every complex elemwise op in chnk.helm2d.green / chnk.helm2d.kern.
    srcinfo = struct('r', rs_flat, 'n', ns_flat);
    targinfo = srcinfo;
    t0 = tic;
    kmat = kern_eval(srcinfo, targinfo);
    fprintf('BENCH: phase=kernel_only t=%.6f\n', toc(t0));
    kernel_fro = norm(kmat, 'fro');

    % ---- Phase: smooth_matrix ----
    % chnk.quadnative.buildmat over (1:nch, 1:nch). Adds weight multiply
    % and reshapes on top of kernel_only.
    wts = chnkr.wstor;
    t0 = tic;
    sysmat_smooth = chnk.quadnative.buildmat(chnkr, kern_eval, opdims, ...
        1:chnkr_nch, 1:chnkr_nch, wts);
    fprintf('BENCH: phase=smooth_matrix t=%.6f\n', toc(t0));
    sysmat_smooth_fro = norm(sysmat_smooth, 'fro');

    % ---- Phase: chunkermat ----
    % Full build: smooth matrix + near/self GGQ corrections.
    t0 = tic;
    sysmat = chunkermat(chnkr, fkern);
    fprintf('BENCH: phase=chunkermat t=%.6f\n', toc(t0));

    % ---- Phase: add_identity ----
    t0 = tic;
    sysmat = 0.5*eye(N) + sysmat;
    fprintf('BENCH: phase=add_identity t=%.6f\n', toc(t0));
    sysmat_fro = norm(sysmat, 'fro');
end
