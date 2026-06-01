% DIAGNOSIS: Element-wise transcendental builtins (exp, sin, cos, tan, log, ...)
% differ at the last ULP between modes. The interpreter (--opt 0) evaluates them
% via the native vectorized libm path (libmvec, built with -ffast-math per the
% CLI --no-fast-math note), which is ~1 ULP off correctly-rounded. The JIT
% (--opt 1) emits unary_kernel(a, Math.exp) / Math.sin etc., i.e. JS Math.*,
% which is correctly rounded here. So the two produce slightly different bits.
% (sqrt agrees in both — IEEE-754 mandates correct rounding for sqrt.)
% Hidden at default display precision; revealed by %.17g.
%
% --opt 0 output:
%   2.7182818284590455
% --opt 1 output:
%   2.7182818284590451
function r = f()
  r = exp([1 2 3]);
end
x=0; for k=1:300; m=f(); x=x+1; end
v = f();
fprintf('%.17g\n', v(1));
