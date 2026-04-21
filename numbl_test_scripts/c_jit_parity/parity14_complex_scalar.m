% C-JIT parity gap #14: complex scalar arithmetic (Phase 1).
%
% The C-JIT historically bailed on any `complex_or_number` scalar with
%   "unsupported type: complex_or_number"
% in src/numbl-core/jit/c/cFeasibility.ts. Phase 1 adds pair-of-doubles
% scalar emission for +, -, *, /, unary -, ImagLiteral, and the
% real / imag / conj builtins.
%
% Expected disp output (should match across all runs):
%   numbl --opt 1 run <this>                         -> pairs below
%   numbl --opt 2 run <this>                         -> pairs below
%   numbl --opt 2 --check-c-jit-parity run <this>    -> pairs below
%   matlab -batch parity14_complex_scalar            -> pairs below

% 1) Literal complex value + arithmetic.
z = 2 + 3i;
w = z * z - 1;               % (2+3i)^2 - 1 = -5 + 12i - 1 = -6 + 12i
disp(real(w))                % -6
disp(imag(w))                % 12

% 2) Division by a complex value.
a = (4 + 2i) / (1 + 1i);     % = (4+2i)(1-i)/2 = (6 - 2i)/2 = 3 - i
disp(real(a))                % 3
disp(imag(a))                % -1

% 3) Conj.
c = conj(1 - 2i);            % = 1 + 2i
disp(real(c))                % 1
disp(imag(c))                % 2

% 4) Complex through a user function, real inputs upcast to complex.
s = complex_sum(5, 7);       % (5 + 7i) + (1i * 2) = 5 + 9i
disp(real(s))                % 5
disp(imag(s))                % 9

% 5) Hot loop exercising the emitter's per-iter pair materialization.
acc = 0 + 0i;
for k = 1:10
    acc = acc + (k + k*1i);  % sum(k + k*i) over k=1..10
end
disp(real(acc))              % 55
disp(imag(acc))              % 55

disp('SUCCESS')

function w = complex_sum(x, y)
    z = x + y * 1i;
    w = z + 1i * 2;
end
