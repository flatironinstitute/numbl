% Regression: a statically single-element slice — v(2:2), M(i, 2:2),
% v(end:end), M(1:1, j) — used in scalar arithmetic or a condition must
% evaluate to a SCALAR, not a 1x1 tensor object.
%
% Before the fix, emitIndexSliceJs (the JS-JIT backend) always returned an
% allocated 1x1 tensor object, even though the lowerer types such a slice
% as a scalar double. Downstream scalar codegen then did object arithmetic:
%   v(2:2) + 1      -> "[object Object]1"   (string concat)
%   M(1, 1:1) * 10  -> NaN
%   if v(2:2) > 15  -> took the WRONG branch (tensorObject > 15 === false)
% The C producer (emitIndex.ts) already unwrapped to a scalar, so --opt 0
% and --opt 2 were correct and only --opt 1 diverged.

%!numbl:assert_jit
v = [10 20 30];
M = [1 2 3; 4 5 6];
acc = 0;
for k = 1:50
    a = v(2:2) + 1;        % 21
    b = v(end:end) * 2;    % 60
    c = M(1, 2:2) * 10;    % 20
    d = M(1:1, 3) + 100;   % 103
    if v(2:2) > 15         % 20 > 15 -> true
        e = 1;
    else
        e = 0;
    end
    acc = acc + a + b + c + d + e;   % 205 per iteration
end
assert(acc == 205 * 50, 'scalar-slice arithmetic/condition wrong');

% A complex single-element slice must yield a complex scalar, and a
% MULTI-element slice must still be a tensor.
C = [1+2i 3+4i 5+6i];
acc2 = 0;
for k = 1:50
    z = C(2:2);            % 3+4i (complex scalar)
    acc2 = acc2 + real(z) + imag(z);  % 7 per iteration
    w = v(1:2) + 1;        % [11 21] (still a tensor)
    acc2 = acc2 + w(2);    % 21 per iteration
end
assert(acc2 == 28 * 50, 'complex scalar slice / multi-element slice wrong');

disp('SUCCESS')
