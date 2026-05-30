% DIAGNOSIS: the C format engine (jit/builtins/runtime/io/format_engine.h)
% mishandles `*` (argument-supplied width/precision). Two defects: (1) the
% walker consumes the VALUE slot before parse_spec consumes the `*` slot, so
% the argument order is reversed; (2) precision `*` (after `.`) is never
% applied. The JS engine (format_engine.js) handles both, so only opt2
% diverges. (All statements suppressed so top-level C-JIT engages.)
%
% --opt 0/1 output (correct):  "3.142" / "   42" / "[3.14]"
% --opt 2 output (buggy):      "  3"   / 42-wide field holding "5" / "  2"
fprintf('%.*f\n', 3, pi);
fprintf('%*d\n', 5, 42);
fprintf('[%.*f]\n', 2, pi);
