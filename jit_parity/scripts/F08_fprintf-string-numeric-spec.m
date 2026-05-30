% TEST: fprintf of a (double-quoted) STRING argument that looks numeric,
% under a numeric conversion: fprintf('%g\n', "12.5").
% opt0 (interp): 12.5
% opt1 (JS-JIT): 12.5
% opt2 (C-JIT):  0     <-- DIVERGES
% DIVERGING MODE: opt2 only (opt0==opt1; MATLAB parses -> 12.5).
%
% Cause: same root as F07 -- the C format engine coerces any TEXT slot to
%   0.0 rather than parsing the numeric string. (A NON-numeric string like
%   "A" is a separate 3-way split: opt0 errors / opt1 NaN / opt2 0 -- the
%   interpreter throws while JS does Number("A")=NaN; that opt0-vs-opt1
%   reconciliation is noted in the fix but this script tracks the clean
%   opt2-only numeric-string case.)
% FIX DIRECTION: C engine should parse a numeric TEXT slot like the JS path.
% JIT engagement: CONFIRMED (void fprintf engages C-JIT).
fprintf('%g\n', "12.5");
