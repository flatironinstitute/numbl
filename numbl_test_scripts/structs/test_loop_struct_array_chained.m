% Stage 13 — chained struct array member read inside a JIT loop.
% Mirrors the chunkie BVH-walk pattern `T.nodes(inode).chld`,
% `T.nodes(inode).xi` where a struct `T` holds a struct array `nodes`
% and each element has scalar + tensor fields that get read (and in
% some cases assigned to a local) inside a tight outer loop.
%
% Lowering: the parser emits `Member(MethodCall(Ident(T), "nodes",
% [i]), "leaf")` for reads in expression position. `lowerExpr` case
% "Member" recognizes this shape when `T` is a struct with a
% `struct_array`-typed field and the leaf field is a scalar numeric
% type or a real tensor, and emits a `StructArrayMemberRead` IR node.
% Codegen hoists `var $T_nodes_elements = T.fields.get("nodes").elements`
% at function entry; per-use reads do
% `$T_nodes_elements[Math.round(i) - 1].fields.get("leaf")`. For
% tensor-typed leaves the result flows into a plain Assign, and the
% existing per-Assign hoist-refresh path then picks up the fresh
% `.data` / `.length` aliases so downstream `chld(k)` reads go through
% the fast scalar-index path.
%
% ``%!numbl:assert_jit`` is placed inside every outer loop body to
% assert the surrounding loop got JIT-compiled — if the marker call
% survives to the interpreter (because lowering bailed), it throws.

% 1) Basic scalar field read across a struct array.
T.nodes(1).val = 10;
T.nodes(2).val = 20;
T.nodes(3).val = 30;
T.nodes(4).val = 40;
T.nodes(5).val = 50;
total = 0;
for i = 1:100
    %!numbl:assert_jit
    inode = mod(i - 1, 5) + 1;
    total = total + T.nodes(inode).val;
end
% Cycle sum = 150; 100/5 * 150 = 3000
assert(total == 3000, '1: scalar field cycle sum');

% 2) Tensor field read assigned to a local, then scalar-indexed.
T2.nodes(1).chld = [7; 8; 9; 10];
T2.nodes(2).chld = [1; 2; 3; 4];
T2.nodes(3).chld = [5; 6; 7; 8];
s2 = 0;
for i = 1:60
    %!numbl:assert_jit
    inode = mod(i - 1, 3) + 1;
    chld = T2.nodes(inode).chld;
    s2 = s2 + chld(1) + chld(4);
end
% Per-cycle: (7+10) + (1+4) + (5+8) = 17 + 5 + 13 = 35
% 60/3 * 35 = 700
assert(s2 == 700, '2: tensor field scalar-indexed');

% 3) Both scalar and tensor fields of the same struct array, used
%    independently inside the same loop.
T3.nodes(1).val = 2;   T3.nodes(1).chld = [100; 200];
T3.nodes(2).val = 3;   T3.nodes(2).chld = [300; 400];
T3.nodes(3).val = 5;   T3.nodes(3).chld = [500; 600];
acc_val = 0;
acc_chld = 0;
for i = 1:30
    %!numbl:assert_jit
    inode = mod(i - 1, 3) + 1;
    acc_val = acc_val + T3.nodes(inode).val;
    chld = T3.nodes(inode).chld;
    acc_chld = acc_chld + chld(1);
end
% val cycle sum = 10; 30/3 * 10 = 100
assert(acc_val == 100, '3: mixed scalar field');
% chld(1) cycle sum = 100 + 300 + 500 = 900; 30/3 * 900 = 9000
assert(acc_chld == 9000, '3: mixed tensor field');

% 4) Nested loops with the outer loop variable driving the index.
T4.nodes(1).val = 1;
T4.nodes(2).val = 2;
T4.nodes(3).val = 3;
T4.nodes(4).val = 4;
grand = 0;
for i = 1:4
    %!numbl:assert_jit
    for j = 1:5
        grand = grand + T4.nodes(i).val * j;
    end
end
% sum(i=1..4, j=1..5) i*j = (sum i) * (sum j) = 10 * 15 = 150
assert(grand == 150, '4: nested loop struct array read');

% 5) Two different fields read in the same iteration (distinct hoists).
T5.nodes(1).a = 1; T5.nodes(1).b = 100;
T5.nodes(2).a = 2; T5.nodes(2).b = 200;
T5.nodes(3).a = 3; T5.nodes(3).b = 300;
sa = 0;
sb = 0;
for i = 1:30
    %!numbl:assert_jit
    inode = mod(i - 1, 3) + 1;
    sa = sa + T5.nodes(inode).a;
    sb = sb + T5.nodes(inode).b;
end
% a cycle sum = 6; 30/3 * 6 = 60
assert(sa == 60, '5: field a');
% b cycle sum = 600; 30/3 * 600 = 6000
assert(sb == 6000, '5: field b');

% 6) Same struct, same field read multiple times per iter (hoist
%    alias is shared; the per-iter elements lookup must stay correct
%    regardless of dedup).
T6.nodes(1).k = 7;
T6.nodes(2).k = 11;
s6 = 0;
for i = 1:20
    %!numbl:assert_jit
    inode = mod(i - 1, 2) + 1;
    s6 = s6 + T6.nodes(inode).k + T6.nodes(inode).k * 2;
end
% each iter contributes (k + 2*k) = 3*k; cycle sum 3*(7 + 11) = 54
% 20/2 * 54 = 540
assert(s6 == 540, '6: repeated field read');

disp('SUCCESS');
