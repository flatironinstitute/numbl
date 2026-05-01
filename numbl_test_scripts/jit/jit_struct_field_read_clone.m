% Regression: a JIT-compiled body that reads a tensor-typed field of a
% struct (or struct-array element) and binds it to a local variable
% must take a value-semantics clone — not alias the field's wrapper
% directly. Otherwise the function-exit dispose pass releases the
% local, which decrements the field's refcount cell to zero and frees
% the buffer the struct still references; a later disposal of the
% struct then double-disposes the field.
%
% Pattern is hot in chunkie's flagnear_rectangle, where a tree of
% struct-array nodes is queried for `T.nodes(i).chld` and `.xi` inside
% a loop, then the function returns. With the borrowed-rhs clone
% missing, the function-exit cleanup of the JIT loop disposed
% `T.nodes(i).chld`, then the outer caller's dispose of the chunker
% (which reaches the same buffers via field aliasing) hit a
% DoubleDisposeError.

function out = consume_field(T)
    % Tensor field of a struct array — must be cloned at this Assign,
    % not aliased.
    chld = T.nodes(1).chld;
    out = sum(chld);
end

T = struct();
node = struct('chld', [10.0; 20.0; 30.0; 40.0]);
T.nodes = node;

s = consume_field(T);
assert(s == 100, 'sum of [10 20 30 40] should be 100');

% T.nodes(1).chld must still hold the buffer (caller invariant).
assert(isequal(T.nodes(1).chld, [10.0; 20.0; 30.0; 40.0]), ...
    'struct field corrupted after callee read');

disp('SUCCESS')
