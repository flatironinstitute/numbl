% C-JIT parity gap #09: TensorLiteral (`[1 2 3]`, `[]`, `[true, false]`).
%
% The JS-JIT compiles TensorLiteral via makeTensor / mkTensor; the
% C-JIT historically bailed feasibility with
%   "unsupported expr: TensorLiteral"
% because tensor-result Assigns had no fresh-buffer path and outputs
% assumed the JS wrapper would pre-allocate a fixed-size buffer. The
% tensor-creation iteration added a dynamic-output ABI (`double **`
% out-param via koffi) so the C function can allocate the buffer
% itself and transfer ownership on return.
%
% Expected disp output (should match across all runs):
%   numbl --opt 1 run <this>                         -> 2\n3\n1\n0\n-1
%   numbl --opt 2 run <this>                         -> 2\n3\n1\n0\n-1
%   numbl --opt 2 --check-c-jit-parity run <this>    -> 2\n3\n1\n0\n-1
%   matlab -batch parity09_tensor_literal            -> 2\n3\n1\n0\n-1

% 1) Row-vector literal from a function with no tensor inputs.
%    Exercises the dynamic-output path: firstTensorLen = 0, so the
%    fixed ABI can't fit the 3 elements — the C malloc + transfer path
%    owns the buffer.
r = make_row(); %#ok<NASGU>
r = make_row(); %#ok<NASGU>
r = make_row();
disp(r(2))      % 2

% 2) 2D literal from function inputs (scalar cells). Column-major order
%    check: literal is [1 2; 3 4] -> data = [1 3 2 4].
M = make_mat(1, 2, 3, 4); %#ok<NASGU>
M = make_mat(1, 2, 3, 4); %#ok<NASGU>
M = make_mat(1, 2, 3, 4);
disp(M(2, 1))   % 3

% 3) Boolean literal. MATLAB stores these as a logical tensor; numbl
%    promotes to a double tensor inside the C-JIT. Value check only.
b = make_bools(); %#ok<NASGU>
b = make_bools(); %#ok<NASGU>
b = make_bools();
disp(b(1))      % 1 (true)
disp(b(2))      % 0 (false)

% 4) Empty literal `[]`: length should be 0, with isempty true.
e = make_empty(); %#ok<NASGU>
e = make_empty(); %#ok<NASGU>
e = make_empty();
assert(isempty(e), 'empty literal should be empty');
disp(-1)

function r = make_row()
    r = [1, 2, 3];
end

function M = make_mat(a, b, c, d)
    M = [a, b; c, d];
end

function b = make_bools()
    b = [true, false, true];
end

function e = make_empty()
    e = [];
end
