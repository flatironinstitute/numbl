% DIAGNOSIS: For a runtime-valued (statically-unknown) size argument, the JIT
% constructor codegen (zeros/ones/...) emits Math.trunc(n) and silently
% truncates a fractional size to an integer. The interpreter instead validates
% that size inputs are nonnegative integers and raises an error. (When the size
% is a compile-time constant the JIT validates and declines, so the divergence
% only surfaces with an opaque runtime value.)
%
% --opt 0 output:
%   RuntimeError ...:4: Size inputs must be nonnegative integers.   (exit=1)
% --opt 1 output:
%   900                                                             (exit=0; 3.7 -> 3)
n = 3.7;
s = 0;
for k=1:300
  a = zeros(1, n);
  s = s + numel(a);
end
disp(s)
