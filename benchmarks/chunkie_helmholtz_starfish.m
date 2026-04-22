function chunkie_helmholtz_starfish()
    % Benchmark: Helmholtz starfish exterior scattering via CFIE.
    % Based on the first example in the chunkie documentation, minus plotting.
    %
    % Written as a function file with a warmup call so numbl's JIT compile /
    % specialization cost is not included in the timed run. The warmup uses a
    % smaller kvec and grid so it hits the same code paths quickly.
    %
    % Emits machine-parseable BENCH: and CHECK: lines so the runner can
    % compare phase timings and result summaries between MATLAB and numbl.
    %
    % See benchmarks/chunkie_helmholtz_starfish.md for full description
    % and typical results.

    t0 = tic;
    mip load --install chunkie;
    fprintf('BENCH: phase=chunkie_load t=%.6f\n', toc(t0));

    % ---- Warmup: small problem, same code paths ----
    % Lands JIT specializations (and the fmm2d / LAPACK bridge) in cache
    % so the timed run below measures pure execution time.
    t_warm = tic;
    [~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = run_bench(5, 0.5, 5*[1;-1.5], 40);
    fprintf('BENCH: phase=warmup t=%.6f\n', toc(t_warm));

    % ---- Hot measurement ----
    t_execution = tic;
    [zk, chnkr_k, chnkr_nch, chnkr_r_norm, sysmat_fro, rhs_norm, ...
     sol_norm, num_exterior, uscat_norm, utot_norm] = ...
        run_bench(5, 0.5, 20*[1;-1.5], 200);
    fprintf('BENCH: phase=execution t=%.6f\n', toc(t_execution));

    % ---- Result checks for cross-implementation comparison ----
    fprintf('CHECK: name=zk value=%.16e\n', zk);
    fprintf('CHECK: name=chnkr_k value=%.16e\n', chnkr_k);
    fprintf('CHECK: name=chnkr_nch value=%.16e\n', chnkr_nch);
    fprintf('CHECK: name=chnkr_r_norm value=%.16e\n', chnkr_r_norm);
    fprintf('CHECK: name=sysmat_fro value=%.16e\n', sysmat_fro);
    fprintf('CHECK: name=rhs_norm value=%.16e\n', rhs_norm);
    fprintf('CHECK: name=sol_norm value=%.16e\n', sol_norm);
    fprintf('CHECK: name=num_exterior value=%.16e\n', num_exterior);
    fprintf('CHECK: name=uscat_norm value=%.16e\n', uscat_norm);
    fprintf('CHECK: name=utot_norm value=%.16e\n', utot_norm);

    fprintf('DONE\n');
    disp('SUCCESS')
end

function [zk, chnkr_k, chnkr_nch, chnkr_r_norm, sysmat_fro, rhs_norm, ...
          sol_norm, num_exterior, uscat_norm, utot_norm] = ...
         run_bench(narms, amp, kvec, grid_n)
    planewave = @(kvec,r) exp(1i*sum(bsxfun(@times,kvec(:),r(:,:)))).';
    zk = norm(kvec);

    % ---- Phase: discretize ----
    t0 = tic;
    chnkr = chunkerfunc(@(t) starfish(t,narms,amp), struct('maxchunklen',4/zk));
    fprintf('BENCH: phase=discretize t=%.6f\n', toc(t0));

    % ---- Phase: build system matrix ----
    t0 = tic;
    fkern = kernel('helm','c',zk,[1,-zk*1i]);
    sysmat = chunkermat(chnkr,fkern);
    sysmat = 0.5*eye(chnkr.k*chnkr.nch) + sysmat;
    fprintf('BENCH: phase=build_matrix t=%.6f\n', toc(t0));

    % ---- Phase: solve ----
    t0 = tic;
    rhs = -planewave(kvec(:),chnkr.r(:,:));
    sol = gmres(sysmat,rhs,[],1e-13,100);
    fprintf('BENCH: phase=solve t=%.6f\n', toc(t0));

    % ---- evaluation targets ----
    x1 = linspace(-3,3,grid_n);
    [xxtarg,yytarg] = meshgrid(x1,x1);
    targets = [xxtarg(:).';yytarg(:).'];

    % ---- Phase: interior test ----
    t0 = tic;
    in = chunkerinterior(chnkr,targets);
    out = ~in;
    fprintf('BENCH: phase=interior t=%.6f\n', toc(t0));

    % ---- Phase: evaluate scattered field at targets ----
    t0 = tic;
    uscat = chunkerkerneval(chnkr,fkern,sol,targets(:,out));
    fprintf('BENCH: phase=eval t=%.6f\n', toc(t0));

    uin = planewave(kvec,targets(:,out));
    utot = uscat(:) + uin;

    chnkr_k = double(chnkr.k);
    chnkr_nch = double(chnkr.nch);
    chnkr_r_norm = norm(chnkr.r(:));
    sysmat_fro = norm(sysmat,'fro');
    rhs_norm = norm(rhs);
    sol_norm = norm(sol);
    num_exterior = double(sum(out(:)));
    uscat_norm = norm(uscat(:));
    utot_norm = norm(utot(:));
end
