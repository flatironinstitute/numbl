% TEST: cat along dim>=3 with an empty operand: cat(3, [1 2;3 4], []).
% MATLAB: drops the empty -> a 2x2 (ndims 2).
% opt0 (interp): RuntimeError "cat: dimension mismatch on dimension 1" <-- DIVERGES
% opt1 (JS-JIT): ndims=2 2x2 (correct)
% opt2 (C-JIT):  ndims=2 2x2 (correct)
% DIVERGING MODE: opt0 only.
%
% Cause: the interpreter cat (dim>=3 path in array-manipulation.ts) pads
%   the [] operand to shape [0,0,1] and runs the cross-dim equality check
%   without first dropping all-empty operands. MATLAB ignores empties in
%   cat. vertcat/horzcat (dim 1/2) already handle empties; only the
%   cat(dim>=3,...) branch is wrong. Repros with cat(3,[],A) and cat(3,5,[]).
% FIX DIRECTION: drop all-empty operands before the dimension-equality
%   check (matching dim 1/2 behavior and MATLAB).
A = [1 2; 3 4];
C = cat(3, A, []);
fprintf('ndims=%d %dx%d\n', ndims(C), size(C,1), size(C,2));
