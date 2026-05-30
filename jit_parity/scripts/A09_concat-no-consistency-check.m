% DIAGNOSIS: JIT-compiled vertical concatenation [a; b] takes the column count
% from the FIRST operand only (_mtoc2_tc = a.shape[1]) and copies each operand
% without verifying subsequent operands have matching column counts. When the
% JIT cannot statically prove the shapes it builds a result sized to the first
% row and silently zero-pads/drops the rest instead of raising
% "Dimensions of arrays being concatenated are not consistent".
%
% --opt 0 output:
%   RuntimeError ...:6: Dimensions of arrays being concatenated are not consistent  (exit=1)
% --opt 1 output:
%   1800                                                                            (exit=0)
n1 = 3; n2 = 2;
s = 0;
for k=1:300
  a = ones(1, n1);
  b = ones(1, n2);
  c = [a; b];
  s = s + numel(c);
end
disp(s)
