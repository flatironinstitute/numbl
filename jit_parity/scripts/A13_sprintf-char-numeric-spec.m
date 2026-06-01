% DIAGNOSIS: sprintf/fprintf with a CHAR argument to a numeric spec
%   (%d, %f, %g, %e, %c, ...) diverges. The interpreter's runtime
%   toNumber(char) returns the char code (charCodeAt -> 65 for 'A'),
%   but the JIT's format engine
%   (src/numbl-core/jit/builtins/runtime/io/format_engine.js, toNumber)
%   uses Number(v.value) = Number("A") = NaN.
%   Stronger variant: a MULTI-char arg ('AB') to %d -> opt0 ERRORS
%   ("Cannot convert multi-char to number"); opt1 prints NaN.
% --opt 0:  [65|65.000000|65]
% --opt 1:  [NaN|NaN|NaN]
function s = f(c)
  s = sprintf('[%d|%f|%g]', c, c, c);
end
n = 0;
for k=1:200          % hot loop -> JIT-compile f
  n = n + length(f('A'));
end
disp(f('A'))         % print the JIT-compiled result once
