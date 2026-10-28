% TEST: cell() with no arguments inside a JIT-compiled loop.
% MATLAB: error MATLAB:minrhs "Not enough input arguments."
% opt0 (interp): the MATLAB error.
% opt1/opt2 (JIT): returned a 0x0 cell and went on   <-- DIVERGED
% FIX: the JIT's cell() lowering declines a zero-argument call, so the
%   interpreter raises the error.
function r = f(n)
  r = 0;
  for i = 1:n
    c = cell();
    r = r + i;
  end
end
disp(f(3));
