# Indexing

Paren and brace indexing on the interpreter path (`A(i,j)`, `A(i) = x`, `A(i) = []`, `c{i}`) enters through `runtime/runtimeIndexing.ts` (`index`, `indexStore`, `indexCell`, `indexCellStore`). Those entry points handle function-handle calls, class `subsref`/`subsasgn`, dictionaries, `end`, and a few fast paths, then hand resolved subscripts to `runtime/indexing.ts`.

## The planner

`runtime/indexPlan.ts` holds the MATLAB indexing rules, independent of how a container stores its elements. Every container is viewed as a flat column-major array plus a shape:

- `planIndex(shape, numel, subs)` plans a read: the selected offsets and the result shape.
- `planStore(shape, numel, subs, rhsNumel, rhsShape)` plans an assignment: the offsets to write, the RHS compatibility check (scalar expansion, element counts for linear indexing, non-singleton dimensions for subscript indexing), and the grown shape when a subscript is out of range.
- `planDelete(shape, numel, subs)` plans `A(subs) = []`: the offsets to keep and the shape afterwards.

The rules applied there include logical masks (as `find(mask)`), colon, unit-step ranges (planned as contiguous blocks), trailing-dimension collapse when there are fewer subscripts than dimensions, the vector-orientation rule for `A(I)`, empty-result shapes, growth (vectors along their orientation, ambiguous growth errors, colon extents taken from the RHS when the target is `0x0`), deletion rules, and MATLAB's error messages.

A plan's selection is either strided (`start + k*step`, used for columns, rows, slabs and ranges) or an explicit offsets array. `forEachGrowRun` and `growsInPlace` describe how existing elements move when storage grows.

## Containers

Each container in `indexing.ts` supplies only gather, scatter and resize on top of the planner:

- Tensors (real, complex, logical) and numeric scalars (treated as 1x1). A 1x1 read result is returned as a scalar; an all-zero imaginary part is dropped; a logical target stays logical and converts assigned values.
- Char arrays (reads; stores and deletion go through the tensor path on character codes). `RuntimeChar` stores rows concatenated, so offsets are mapped through (row, column).
- Cell arrays (paren and brace reads and stores, deletion, N-d). Growth pads with `[]`.

Sparse matrices, string arrays, struct arrays and class-instance arrays still resolve subscripts with their own code; moving them onto the planner means giving each a gather/scatter/resize over its element array (and, for struct arrays, a shape).
