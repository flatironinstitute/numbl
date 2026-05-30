% DIAGNOSIS: isfinite() of a logical scalar diverges. JIT emits
%   Number.isFinite(x) on the raw JS boolean; Number.isFinite(true)===false
%   (no numeric coercion), so JIT returns 0. Interpreter coerces true->1
%   (finite) and returns 1.
%   Root: src/numbl-core/jit/builtins/defs/logical/isinf.ts isfinite.jsScalar
% --opt 0:  10000   (isfinite(true)==1)
% --opt 1:  0       (isfinite(true)==0)
function r = f(x)
  r = 0;
  for k=1:200
    r = r + isfinite(x);
  end
end
s = 0;
for k=1:50
  s = s + f(true);
end
disp(s)
