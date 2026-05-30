% TEST: a construct with a RUNTIME (non-statically-known) dimension that
% activates an inline runtime snippet — `ones(1, n)` (CHECK_DIM_SNIPPET,
% the dim validator) and a runtime-shaped vertcat `[a; b]`
% (CONCAT_CHECK_SNIPPET, the consistency guard).
% opt0 (interp):  prints "4 8"   (correct)
% opt1 (JS-JIT):  prints "4 8"   (correct, matches)
% opt2 (C-JIT):   RuntimeError: mtoc2-c-jit compile failed
%                   error: #include expects "FILENAME" or <FILENAME>
%                          #include math.h
% DIVERGES: opt2 errors while opt0/opt1 succeed.
% CAUSE: two InlineSnippet objects declared their `headers` WITHOUT the
%   bracket delimiters (`["math.h", ...]` instead of `["<math.h>", ...]`):
%   CHECK_DIM_SNIPPET in builtins/defs/shape/_construct.ts and
%   CONCAT_CHECK_SNIPPET in codegen/emitTensorConcat.ts. emit.ts renders
%   each header verbatim as `#include ${h}`, so a bare `math.h` emits the
%   invalid `#include math.h`. File-loaded snippets get the brackets for
%   free from parseSnippetSource; these two hand-rolled ones shipped the
%   unbracketed form, and the duplicate failed to dedupe against the
%   bracketed BASE_HEADERS. Fixed by bracketing both header arrays;
%   collectRuntimeHeaders now also asserts every header is bracketed/quoted.
% JIT-ENGAGEMENT: confirmed — opt2 reaches the C compiler (the failure was
%   IN the generated C); `%!numbl:opaque n` keeps the dim dynamic so the
%   validator/concat-guard snippets are actually emitted.
n = 4;
%!numbl:opaque n
a = ones(1, n);
b = ones(1, n);
M = [a; b];
fprintf('%d %d\n', numel(a), numel(M));
