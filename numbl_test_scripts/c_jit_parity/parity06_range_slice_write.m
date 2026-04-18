% C-JIT parity gap #06: AssignIndexRange (dst(a:b) = src(c:d)).
%
% The JS-JIT compiles `dst(a:b) = src(c:d)` via setRange1r_h; the
% C-JIT historically bailed feasibility with
%   "unsupported stmt: AssignIndexRange"
% because numbl_jit_runtime had no range-write helper.
%
% Expected disp output (should match across all runs):
%   numbl --opt 1 run <this>                         -> 42\n11\n-99\n7
%   numbl --opt 2 run <this>                         -> 42\n11\n-99\n7  (silent JS-JIT fallback)
%   numbl --opt 2 --check-c-jit-parity run <this>    -> 42\n11\n-99\n7  (the gap is now closed)
%   matlab -batch parity06_range_slice_write         -> 42\n11\n-99\n7
%
% Also asserts that the caller's `A` is unchanged after the range
% write — the unshare-at-entry path (parity03) has been extended to
% cover AssignIndexRange targets.

% 1) Basic range copy into a pure-input tensor param — caller's A
%    must stay untouched.
A = [1 2 3 4 5 6 7 8 9 10];
B = range_copy(A, [42 43 44]);
assert(isequal(A, [1 2 3 4 5 6 7 8 9 10]), 'caller A must be unchanged');
disp(B(2))

% 2) Whole-tensor RHS form: dst(a:b) = src (no explicit range on src).
%    Exercises the srcStart/srcEnd = null branch of AssignIndexRange.
C = zeros(1, 10);
src = [11 22 33 44 55];
D = whole_tensor_copy(C, src);
disp(D(3))

% 3) Length-mismatch should throw at runtime with the exact MATLAB
%    error message (err-flag code 3.0 in the C path).
threw = false;
try
    D2 = range_copy_bad(zeros(1, 10));
catch
    threw = true;
end
assert(threw, 'length-mismatch should have thrown');
disp(-99)

% 4) Overlapping self-copy: memmove in the C helper must handle
%    same-buffer overlap.
E = [1 2 3 4 5 6 7 8];
F = self_copy(E);
disp(F(3))

function B = range_copy(B, v)
    B(2:4) = v(1:3);
end

function D = whole_tensor_copy(D, src)
    D(3:7) = src;
end

function D = range_copy_bad(D)
    tmp = [1 2];
    D(1:5) = tmp(1:2);  % size mismatch: 5 != 2
end

function F = self_copy(F)
    % shift tail forward: F(1:4) = F(5:8)
    F(1:4) = F(5:8);
end
