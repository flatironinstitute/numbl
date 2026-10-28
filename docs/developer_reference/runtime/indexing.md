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
- Struct arrays (N-d `shape`; a scalar struct is indexed as 1x1). A one-element result is the element itself. `s(k) = t` requires `t` to have the target's fields (any order); the store-back of `s(k).f = v` may add fields, which every element then gets. Growth pads with structs whose fields are `[]`. An empty numeric target becomes a struct array.
- Class-instance arrays (2-D; a scalar instance is indexed as 1x1, after any `subsref`/`subsasgn` overload has had its turn in `runtimeIndexing.ts`). Growth pads with instances whose properties are `[]`.
- String arrays (2-D; a 1x1 result is the primitive string). Growth pads with `""` (numbl has no `<missing>`). Brace reads return char vectors.
- Sparse matrices keep CSC storage. With one subscript they walk the plan's linear offsets; with two or more they use the per-dimension forms `planIndexDims` / `planStoreDims` / `planDeleteDims`, so selecting rows and columns never enumerates the dense size. Stores collect (row, column, value) triples and merge them into the columns (later writes win, zeros remove entries). Results stay sparse, except that a one-element read returns a number (MATLAB returns a 1x1 sparse matrix).

`runtimeIndexing.ts` keeps fast paths that skip the planner for one in-range scalar subscript: reads from tensors, struct arrays, object arrays and string arrays, writes into unshared real tensors (and `v(end+1) = x`), and writes of an element into an unshared struct array (same fields, same order) or object array (same class).
