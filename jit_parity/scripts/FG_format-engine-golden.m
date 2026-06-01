% GOLDEN VECTORS: pins the printf/fprintf format engine across all three
% modes. fprintf at opt1 runs format_engine.js; at opt2 it runs
% format_engine.h — two full, hand-synced reimplementations that have
% repeatedly drifted (the entire F-class: F01..F08). This script exercises
% the conversion surface that ALL THREE modes currently agree on, so any
% future edit that desyncs the two engines on a previously-agreeing case
% trips the gate immediately.
%
% This is an all-fprintf (void-call) script with no unsuppressed output, so
% the whole scope JIT-compiles; --dump-c confirms the C engine is exercised
% (~1300 lines of C), i.e. opt1/opt2 really go through the format engines
% rather than the interpreter.
%
% DELIBERATELY EXCLUDED (known divergences, tracked for the hunting phase —
% do NOT add vectors for these here or the gate will fail):
%   - the printf space flag on a numeric spec ('% d', '% f'): opt0+opt1
%     ignore it, opt2 honors it (MATLAB honors it — the JS family is wrong).
%   - non-ASCII '%c' (opt2 emits UTF-8 bytes; opt1 a UTF-16 code unit).
%   - exact-halfway %f/%e/%g rounding and non-integer %s shortest round-trip
%     (the dtoa cases already excluded as F01/F02/F04).

% --- %d / %i / %u: integers, negatives, big integer, non-finite ---
fprintf('%d %d %d\n', 0, -7, 42);
fprintf('%i|%u\n', -13, 99);
fprintf('%5d|%-5d|%05d\n', 7, 7, 7);
fprintf('%+d\n', 5);
fprintf('%d\n', 1e19);
fprintf('%d %d\n', 1/0, -1/0);
fprintf('%d\n', 0/0);

% --- %f / %e / %g: non-tie values, width/precision/flags, non-finite ---
fprintf('%f|%.2f|%.0f\n', 3.14159, 3.14159, 12.3);
fprintf('%8.3f|%-8.3f\n', 2.71828, 2.71828);
fprintf('%e|%E\n', 12345.678, 0.00012345);
fprintf('%.3e\n', 6.022e23);
fprintf('%g|%G\n', 0.0001, 1234567);
fprintf('%g\n', 100000);
fprintf('%g %g\n', 1/0, 0/0);

% --- %x / %X / %o: positive + negative (rounding agrees post-F05) ---
fprintf('%x|%X|%o\n', 255, 255, 64);
fprintf('%x\n', -16);

% --- %s / %c (ASCII), char/string args under a numeric spec (F07/F08) ---
fprintf('%s|%s\n', 'hello', "world");
fprintf('%c%c%c\n', 65, 66, 67);
fprintf('%d\n', 'A');
fprintf('%g\n', "12.5");

% --- literals, %%, escapes ---
fprintf('%5.2f%%\n', 50.0);
fprintf('a\tb\\c\n');
