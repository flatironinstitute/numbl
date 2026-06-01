% TEST: repmat with a ZERO replication count on a multi-element block:
% repmat([1 2; 3 4], 0, 2).
% MATLAB: a 0x4 empty.
% opt0 (interp): RuntimeError "offset is out of bounds"   <-- DIVERGES (crash)
% opt1 (JS-JIT): 0x4 (correct)
% opt2 (C-JIT):  0x4 (correct)
% DIVERGING MODE: opt0 only (the interpreter is the broken one).
%
% Cause: the interpreter repmat tiling loop (interpreter/builtins/
%   array-manipulation.ts) only special-cases rep===1; for rep===0 it
%   allocates a zero-length buffer but the general branch still does
%   newData.set(curData.subarray(...), dstBase) writing blockSize (>1)
%   elements into the empty buffer -> out-of-bounds. Triggers when a 0 rep
%   applies to a block with blockSize>1: repmat(A,0,2), repmat(v,2,0),
%   repmat(A,[2 2 0]).
% FIX DIRECTION: handle rep===0 (skip the copy / produce the empty result).
A = [1 2; 3 4];
C = repmat(A, 0, 2);
fprintf('%dx%d\n', size(C,1), size(C,2));
