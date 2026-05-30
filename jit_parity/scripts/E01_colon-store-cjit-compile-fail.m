% TEST: single-slot Colon store `v(:)=7` as the program's only indexed store.
% opt0 (interpreter): prints "7   7   7"   (correct)
% opt1 (JS-JIT):      prints "7   7   7"   (correct, matches)
% opt2 (C-JIT):       RuntimeError: mtoc2-c-jit compile failed
%                       error: implicit declaration of function 'setjmp'
%                       error: 'mtoc2_grow_bail_buf' undeclared
% DIVERGES: opt2 (errors while opt0/opt1 succeed -> counts as divergence)
% CAUSE: emit.ts bodyHasIndexStore() flags ANY IndexSliceStore -> arms the
%   host-entry setjmp(mtoc2_grow_bail_buf) guard (emitStmt.ts:307). But the
%   Colon/multi-slot IndexSliceStore codegen paths in emitIndex.ts never call
%   useRuntimeByName(state,"mtoc2_grow_bail"), so grow_bail.h (the jmp_buf decl
%   + #include <setjmp.h>) is never activated -> C fails to compile. Only the
%   scalar IndexStore and single-slot Range store paths activate grow_bail.
% JIT-ENGAGEMENT: confirmed - opt1 dump-js non-empty (eng=231); opt2 reaches
%   the C compiler (the failure IS in the generated C, dump-c=548+ lines).
v=[1 2 3];
v(:)=7;
disp(v);
