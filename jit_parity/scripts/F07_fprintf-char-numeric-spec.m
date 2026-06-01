% TEST: fprintf of a CHAR argument under numeric / %c conversions:
% fprintf('%d',  'A')  -> MATLAB 65 (char code)
% fprintf('[%c]','X')  -> MATLAB [X]
% opt0 (interp): 65 / [X]
% opt1 (JS-JIT): 65 / [X]
% opt2 (C-JIT):  0  / [<NUL>]   <-- DIVERGES (a NUL byte for %c)
% DIVERGING MODE: opt2 only.
%
% Cause: the C format engine (io/format_engine.h mtoc2__format_walk) reads
%   `double dval = (slot.kind==MTOC2_FA_DOUBLE)?slot.d:0.0;` so ANY char/
%   string (MTOC2_FA_TEXT) slot fed to %d/%i/%u/%f/%e/%g/%x/%o/%c is
%   silently coerced to 0.0 instead of being converted (single char ->
%   code point). %c of 0.0 then writes a literal NUL byte.
% FIX DIRECTION: in the C engine, convert a TEXT slot reaching a numeric/%c
%   conversion to a number the way opt0/opt1 do (char -> code point),
%   matching the interpreter/JS path and MATLAB.
% JIT engagement: CONFIRMED (void fprintf engages C-JIT; ~1100-line dump).
fprintf('%d\n', 'A');
fprintf('[%c]\n', 'X');
