% TEST: disp() of a struct whose fields are NOT in alphabetical order:
% struct('width',10,'height',20,'depth',5).
% MATLAB: fields display in INSERTION order (width, height, depth).
% opt0 (interp): width:10 / height:20 / depth:5   (correct)
% opt1 (JS-JIT): depth:5 / height:20 / width:10   <-- DIVERGES (alphabetical)
% opt2 (C-JIT):  depth:5 / height:20 / width:10   <-- DIVERGES (alphabetical)
% DIVERGING MODE: opt1 and opt2 (both reorder; insidious -- looks valid).
%
% Cause: structType() in jit/lowering/types.ts canonicalizes a struct's
%   fields by sorting on name. The C _disp (emitNamedTypedef.ts) and the
%   JS-JIT struct object are both built in that sorted order, so display
%   order is silently alphabetical.
% FIX DIRECTION: keep the sort for type IDENTITY/unification, but preserve
%   the original insertion order for DISPLAY (track original order on the
%   struct type and have the disp helpers walk it).
% JIT engagement: CONFIRMED (1-iter loop forces compile; C-JIT engaged).
for i = 1:1
  cfg = struct('width', 10, 'height', 20, 'depth', 5);
  disp(cfg);
end
