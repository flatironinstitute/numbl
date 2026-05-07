% COW correctness for cell- and struct-field writes inside JIT-able
% for loops. Each pattern stresses a refcount-tracking path that the
% JIT compiles into hot inline code. Failure modes:
%   - JIT cell-write skipping incref/decref → caller's tensor sees
%     post-loop mutation through aliased cell elements.
%   - JIT struct-field-set bypassing bindField → struct's _destroy
%     decrefs values that were never incref'd → underflow.
%   - JIT call boundary not increfing args → callee's mutation through
%     a parameter corrupts the caller's binding.

% ── 1. Cell-element loop populates from a shared tensor ─────────────
master = [1 2 3 4 5];
copies = cell(1, 3);
for k = 1:3
    copies{k} = master;
end
master(1) = 999;
for k = 1:3
    assert(copies{k}(1) == 1, sprintf('1: copies{%d}(1) corrupted', k));
end
assert(master(1) == 999, '1: master mutated');

% ── 2. Loop builds struct via field-set, passes to handle ───────────
% The function-handle call site is what exercises strict refcount on
% the JIT struct-field path (the handle's body decrefs the struct's
% fields when the struct is destroyed).
h = @(s) s.x + s.y;
acc = 0;
for i = 1:10
    s = [];
    s.x = i * 1.0;
    s.y = i * 2.0;
    acc = acc + h(s);
end
expected = 0;
for i = 1:10
    expected = expected + i * 1.0 + i * 2.0;
end
assert(abs(acc - expected) < 1e-12, sprintf('2: acc expected %g, got %g', expected, acc));

% ── 3. Loop builds struct with tensor fields, passes to handle ──────
ht = @(s) sum(s.r);
base = [1.0; 2.0; 3.0; 4.0];
acc = 0;
for i = 1:5
    s = [];
    s.r = base * i;
    acc = acc + ht(s);
end
expected = 0;
for i = 1:5
    expected = expected + sum(base * i);
end
assert(abs(acc - expected) < 1e-9, sprintf('3: acc expected %g, got %g', expected, acc));

% ── 4. Function-arg COW: callee mutates parameter ───────────────────
function out = mutate_cell_helper(c)
    c{2} = 999;
    out = c;
end
c = {1, 2, 3};
d = mutate_cell_helper(c);
d = mutate_cell_helper(c);
d = mutate_cell_helper(c);
assert(c{2} == 2, '4: caller c{2} unchanged');
assert(d{2} == 999, '4: returned d{2} mutated');

% ── 5. Function-arg COW: callee mutates struct field ────────────────
function out = mutate_struct_helper(s)
    s.x = 555;
    out = s;
end
s.x = 10;
s.y = 20;
t = mutate_struct_helper(s);
t = mutate_struct_helper(s);
t = mutate_struct_helper(s);
assert(s.x == 10, '5: caller s.x unchanged');
assert(t.x == 555, '5: returned t.x mutated');

% ── 6. Function-arg COW: callee mutates tensor ──────────────────────
function out = mutate_tensor_helper(v)
    v(3) = 99;
    out = v;
end
v = [1 2 3 4 5];
y = mutate_tensor_helper(v);
y = mutate_tensor_helper(v);
y = mutate_tensor_helper(v);
assert(v(3) == 3, '6: caller v(3) unchanged');
assert(y(3) == 99, '6: returned y(3) mutated');

disp('SUCCESS')
