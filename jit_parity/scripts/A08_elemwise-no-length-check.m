% DIAGNOSIS: When the JIT cannot statically prove two 1xN tensor operands have
% equal length, it emits the same-shape kernel (mtoc2_tensor_plus_tt / tt_kernel)
% which allocates the result with a.shape and reads b.data[i] with NO runtime
% length check. Mismatched lengths read out-of-bounds (undefined -> NaN) instead
% of raising the "Matrix dimensions must agree" error the interpreter raises.
%
% --opt 0 output:
%   RuntimeError at ...:6: Matrix dimensions must agree: [1,3] vs [1,2]   (exit=1)
% --opt 1 output (pre-fix):
%   NaN                                                                   (exit=0)
% --opt 2 had the SAME gap on the C side: the fused inline loop
%   (emitTensorFused.ts) and the `_tt` macro (tensor_elemwise_real.h) read
%   past the shorter operand. It was long masked by an unrelated C-compile
%   bug (unbracketed `#include math.h`, see E06) that made opt2 error too;
%   once that was fixed, opt2 produced a garbage sum. Fixed by (a) a
%   runtime shape guard in the fused emitter that falls back to the bcast
%   helper, and (b) a shapes-equal check in `_tt` plus a broadcast-
%   compatibility check in `_bcast_tt` — mirroring the JS tt_kernel.
n1 = 3; n2 = 2;
s = 0;
for k=1:300
  a = ones(1, n1);
  b = ones(1, n2);
  c = a + b;
  s = s + sum(c(:));
end
disp(s)
