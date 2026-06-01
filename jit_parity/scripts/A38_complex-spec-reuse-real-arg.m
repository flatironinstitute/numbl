% TEST: a function first called with a complex scalar (compiling a
% complex_or_number specialization), then called with a REAL scalar that
% reuses that specialization.
% opt0 (interp): 4 + 2i / 6
% opt1 (JS-JIT): 4 + 2i / NaN + NaNi   <-- DIVERGES (real arg corrupted)
% opt2 (C-JIT):  4 + 2i / 6
% DIVERGING MODE: opt1 only (opt0==opt2). MATLAB gold = 6.
%
% Cause: type-widening collapses a real (number)/boolean scalar arg into
%   complex_or_number and reuses the previously-compiled complex spec, but
%   the JS value adapter numblToJit (executors/jit/valueAdapter.ts) is not
%   type-aware: it passes a real scalar as a bare JS number, so the
%   complex-typed body reads .re/.im off a number -> undefined -> NaN. The
%   C adapter (valueAdapterC.ts) boxes a real as {re, im:0}, so opt2 is
%   correct. Trigger needs per-statement dispatch (the try;catch;end forces
%   it). Also repros with a pure-imaginary first arg.
% JIT engagement: CONFIRMED (both JITs hit the complex spec; only JS wrong).
function r = twice(x)
  r = x + x;
end
try;catch;end
disp(twice(2+1i))
disp(twice(3))
