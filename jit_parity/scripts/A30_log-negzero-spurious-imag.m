% TEST: log/log2/log10 of negative zero (runtime -0) under the real->complex lift.
% The JIT statically lifts log(x) of an unknown-sign runtime x to the complex
% path. The JS runtime clog({re:-0, im:0}) computes im = atan2(0,-0) = pi, so
% log(-0) wrongly becomes a complex number. The opt0 interpreter decides the
% lift per-VALUE at runtime: -0 is not < 0, so it takes the REAL branch ->
% real -Inf. The C runtime accidentally matches opt0 because mtoc2_cmake(-0,0)
% (= `-0.0 + 0.0*I`) collapses -0 to +0 before clog.
%
% opt0:  -Infinity            /  -Infinity   (real, reference)
% opt1:  -Infinity + 3.1416i  /  -Infinity   (DIVERGES: spurious complex for -0)
% opt2:  -Infinity            /  -Infinity   (matches opt0 when C-JIT runs;
%                                              == opt1 when C-JIT falls back to JS)
% DIVERGES: opt1 (always); opt2 only on JS fallback.
% ROOT CAUSE: JS clog (cscalar.js / _complex_fold.ts cLog) uses atan2(im,re)
%   with re=-0 -> pi; the lift should not fire for a value that is >= 0 at
%   runtime, but the JIT decides statically. C cmake collapses -0 so masks it.
%!numbl:assert_jit
function test()
  s = [-1, 1];
  z = [0, 0];
  for k = 1:2
    x = s(k) * z(k);   % runtime -0 (k=1), +0 (k=2); not foldable
    disp(log(x));
  end
end
test()
