% DIAGNOSIS: When the JIT cannot statically prove two 1xN tensor operands have
% equal length, it emits the same-shape kernel (mtoc2_tensor_plus_tt / tt_kernel)
% which allocates the result with a.shape and reads b.data[i] with NO runtime
% length check. Mismatched lengths read out-of-bounds (undefined -> NaN) instead
% of raising the "Matrix dimensions must agree" error the interpreter raises.
%
% --opt 0 output:
%   RuntimeError at ...:6: Matrix dimensions must agree: [1,3] vs [1,2]   (exit=1)
% --opt 1 output:
%   NaN                                                                   (exit=0)
n1 = 3; n2 = 2;
s = 0;
for k=1:300
  a = ones(1, n1);
  b = ones(1, n2);
  c = a + b;
  s = s + sum(c(:));
end
disp(s)
