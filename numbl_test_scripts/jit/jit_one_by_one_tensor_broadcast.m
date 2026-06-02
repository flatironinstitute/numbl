% Regression: a genuine 1x1 tensor used in an element-wise op against a
% multi-element tensor must broadcast like a scalar, not yield NaN.
%
% The compiler collapses a [1,1] tensor type to a scalar double/complex
% (!isMultiElement), so the fused/elementwise codegen emits `a.data[i] + b`
% expecting `b` to be a scalar. But the value adapters marshaled the 1x1
% RuntimeTensor as a tensor OBJECT, so `number + object` produced NaN at
% both --opt 1 (numblToJit) and --opt 2 (marshalOneInput). The fix unwraps
% a 1x1 tensor to its element when the compiled param is scalar.
%
% The 1x1 tensor must be a LOOP INPUT for the bug to bite — whole-program
% compilation folds it to a literal. The try/catch forces the top-level
% executor to decline so the jit-loop executor compiles the loop and
% captures `b`/`cb` as marshaled inputs.

b = ones(1, 1) .* 5;          % a real 1x1 tensor (NOT a scalar number)
cb = ones(1, 1) .* (2 + 3i);  % a complex 1x1 tensor
a = [1 2 3];
try
    q = 1;
catch
    q = 2;
end

total = 0;
for k = 1:100
    %!numbl:assert_jit
    s1 = sum(a + b);          % broadcast add  -> 21
    s2 = sum(a .* b);         % broadcast mul  -> 30
    s3 = sum(b - a);          % reversed order -> (5-1)+(5-2)+(5-3) = 9
    z = sum(a + cb);          % complex broadcast -> 12 + 9i
    total = total + s1 + s2 + s3 + real(z) + imag(z);  % 21+30+9+12+9 = 81
end
assert(total == 81 * 100, '1x1-tensor broadcast wrong');

disp('SUCCESS')
