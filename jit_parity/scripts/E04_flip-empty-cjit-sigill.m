% TEST: fliplr/flip of a tensor that is empty along a dimension BELOW the
% flip axis (e.g. fliplr(zeros(0,3)) -> a 0x3 result).
% opt0 (interp): 0x3 / DONE
% opt1 (JS-JIT): 0x3 / DONE
% opt2 (C-JIT):  (no output; process killed, SIGILL / exit 132)  <-- DIVERGES
% DIVERGING MODE: opt2 only (hard crash).
%
% Cause: tensor_ops/tensor_flip.h mtoc2_tensor_flip computes
%   numOuter = total / slabSize, and slabSize comes from a stride that is 0
%   when a lower dim has size 0 (total==0 too) -> 0/0 integer division ->
%   SIGILL. The axisSize<=1 early-return doesn't cover this. Needs a
%   total==0 / slabSize==0 guard. The _complex sibling has the same divide.
%   Triggers: fliplr(0xN), flip(0xN,2), flip(zeros(2,0,3),3).
% JIT engagement: CONFIRMED (C-JIT engaged; crash in the generated kernel).
A = zeros(0,3);
B = fliplr(A);
fprintf('%dx%d\n', size(B,1), size(B,2));
disp('DONE');
