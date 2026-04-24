function run_all()
    % Call each bench 4 times so JIT/compile-cache warm-up is observable.
    fprintf('--- bench_01_scalar_accum ---\n');
    for r = 1:4; fprintf('[call %d] ', r); bench_01_scalar_accum(); end
    fprintf('--- bench_02_scalar_locals ---\n');
    for r = 1:4; fprintf('[call %d] ', r); bench_02_scalar_locals(); end
    fprintf('--- bench_03_scalar_builtins ---\n');
    for r = 1:4; fprintf('[call %d] ', r); bench_03_scalar_builtins(); end
    fprintf('--- bench_04_tensor_elem_read ---\n');
    for r = 1:4; fprintf('[call %d] ', r); bench_04_tensor_elem_read(); end
    fprintf('--- bench_05_tensor_elem_write ---\n');
    for r = 1:4; fprintf('[call %d] ', r); bench_05_tensor_elem_write(); end
    fprintf('--- bench_06_inline_elemwise ---\n');
    for r = 1:4; fprintf('[call %d] ', r); bench_06_inline_elemwise(); end
    fprintf('--- bench_07_chained_elemwise_reduce ---\n');
    for r = 1:4; fprintf('[call %d] ', r); bench_07_chained_elemwise_reduce(); end
    fprintf('--- bench_08_reduction_feedback ---\n');
    for r = 1:4; fprintf('[call %d] ', r); bench_08_reduction_feedback(); end
end
