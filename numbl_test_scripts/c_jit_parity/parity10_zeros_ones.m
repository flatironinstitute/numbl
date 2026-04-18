% C-JIT parity gap #10: zeros / ones with numeric shape args.
%
% The JS-JIT dispatches `zeros(n, m)` / `ones(n, m)` through their
% interpreter builtins; the C-JIT historically bailed feasibility with
%   "non-C-mappable builtin: zeros"
% because these allocate a fresh buffer whose size isn't tied to any
% input tensor — the fixed-output ABI has no slot for it. This gap
% closes via the dynamic-output ABI (shared with TensorLiteral) plus
% inline `calloc` / malloc-and-fill codegen.
%
% Expected disp output (should match across all runs):
%   numbl --opt 1 run <this>                         -> 6\n9\n8\n42\n4\n16
%   numbl --opt 2 run <this>                         -> 6\n9\n8\n42\n4\n16
%   numbl --opt 2 --check-c-jit-parity run <this>    -> 6\n9\n8\n42\n4\n16
%   matlab -batch parity10_zeros_ones                -> 6\n9\n8\n42\n4\n16

% 1) zeros(n, m) with runtime n, m — exercises the dynamic-output ABI
%    and the col-major fill. Verify shape and sum.
Z = make_zeros(2, 3); %#ok<NASGU>
Z = make_zeros(2, 3); %#ok<NASGU>
Z = make_zeros(2, 3);
disp(numel(Z))                % 6
assert(isequal(size(Z), [2 3]), 'zeros shape');
assert(sum(Z(:)) == 0, 'zeros sum');

% 2) ones(n): square-matrix form. n=3 -> 3x3 matrix of 1s, numel = 9.
A = make_ones(3); %#ok<NASGU>
A = make_ones(3); %#ok<NASGU>
A = make_ones(3);
assert(isequal(size(A), [3 3]), 'ones(3) shape');
disp(numel(A))                % 9
assert(A(2, 2) == 1, 'ones(3)(2,2)');

% Also print 8 to separate phases.
disp(8)

% 3) Round-trip through AssignIndex on a fresh zeros: out_pt = zeros(5, 1);
%    out_pt(3) = 42. Verify the write land on the C-owned buffer.
out = zeros_then_write(); %#ok<NASGU>
out = zeros_then_write(); %#ok<NASGU>
out = zeros_then_write();
assert(isequal(size(out), [5 1]), 'out shape');
disp(out(3))                  % 42
assert(out(1) == 0 && out(2) == 0 && out(4) == 0 && out(5) == 0, 'other 0');

% 4) Runtime reassign: out_pt starts zeros(4, 1), grows to zeros(16, 1).
%    The C code must free-and-malloc a new buffer and update shape.
out2 = zeros_reassign(); %#ok<NASGU>
out2 = zeros_reassign(); %#ok<NASGU>
out2 = zeros_reassign();
assert(isequal(size(out2), [16 1]), 'out2 shape');
disp(out2(1))                 % 4 (we set out2(1) = 4 after the reassign)
disp(16)                      % total length sanity

function Z = make_zeros(n, m)
    Z = zeros(n, m);
end

function A = make_ones(n)
    A = ones(n);
end

function out = zeros_then_write()
    out = zeros(5, 1);
    out(3) = 42;
end

function out = zeros_reassign()
    out = zeros(4, 1); %#ok<PREALL>
    out = zeros(16, 1);
    out(1) = 4;
end
