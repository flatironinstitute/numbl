% TEST: sum(A, dim) with dim > ndims(A) on a NON-exact tensor.
% In MATLAB/numbl, reducing along a trailing singleton axis is the
% identity: sum([1 2;3 4], 3) == [1 2;3 4].
% A is passed as a function parameter so its data is NOT statically
% exact -> the compile-time fold is bypassed and the runtime helper
% mtoc2_sum_dim is emitted (confirmed via --dump-js / --dump-c).
%
% OUTPUTS:
%   opt0 (interp): [1 2; 3 4]   <- correct
%   opt1 (JS-JIT): [0 0; 0 0]   <- WRONG (diverges)
%   opt2 (C-JIT):  [1 2; 3 4]   <- correct
%
% DIVERGING MODE: opt1 only (opt0 == opt2).
% JIT ENGAGEMENT: confirmed — dump-js & dump-c both emit
%   mtoc2_sum_dim(A, 3); dump-c is non-empty (C path engaged).
%
% CAUSE: JS runtime accum_dim() (tensor_reduce_real.js, dim>ndim branch)
% returns mtoc2_tensor_alloc_nd(...) WITHOUT copying t.data — a
% zero-filled tensor. Its own comment admits "we don't memcpy though".
% The C macro MTOC2_DEFINE_ACCUM_REDUCTION (tensor_reduce_real.h) does
% memcpy the data, so C is correct. Fix: out.data.set(t.data) in the
% accum_dim dim>ndim branch. Affects sum/prod/mean (shared helper).
function r = f(A)
  r = sum(A, 3);
end
M = [1 2; 3 4];
disp(f(M));
