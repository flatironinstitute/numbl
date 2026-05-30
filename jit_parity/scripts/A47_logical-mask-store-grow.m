% TEST: a logical-mask STORE whose truthy bit is past the end of the array:
% x = [1 2 3]; x(logical([0 0 0 1])) = 9.
% MATLAB: GROWS the array -> x = [1 2 3 9].
% opt0 (interp): 1 2 3        <-- WRONG (drops the out-of-range truthy bit)
% opt1 (JS-JIT): 1 2 3        <-- WRONG (same)
% opt2 (C-JIT):  error "Index exceeds array bounds"  <-- WRONG (errors)
% DIVERGING MODE: all three are wrong AND diverge from each other.
%
% Cause: none of the three implement logical-mask grow-on-store. opt0/opt1
%   silently no-op; C-JIT bounds-aborts.
% FIX DIRECTION: at minimum make the three AGREE. Since grow-on-store via a
%   logical mask is hard to JIT, the chosen direction is to DECLINE this
%   store shape to the interpreter and (separately) make the interpreter
%   grow correctly. Tracked here so the divergence stays visible.
x = [1 2 3];
x(logical([0 0 0 1])) = 9;
disp(x);
