% A struct crossing the JIT boundary.
%
% Structs used to be rejected by the type adapter, so a parameter struct —
% the `p.tol` idiom that pervades numerical MATLAB — declined the enclosing
% loop or call before lowering even started. A struct whose fields are all
% themselves supported now maps to the compiler's StructType.
%
% What must hold: field reads, tensor and char fields, structs returned from
% a JIT'd function, nested structs, and pass-by-value (a callee's write to a
% field must not be visible to the caller).

p = struct('tol', 0.5, 'n', 4, 'v', [1 2 3], 'name', 'abc');

total = 0;
for k = 1:10
    %!numbl:assert_jit
    total = total + use(p, k);
end
assert(abs(total - (0.5 * 55 + 40 + 60)) < 1e-12, 'struct field reads');

q = build(2.5);
assert(q.a == 2.5 && q.b == 5, 'struct returned from a spec');

r = struct('inner', struct('z', 7), 'w', 2);
assert(deep(r) == 14, 'nested struct field read');

p2 = bump(p);
assert(p2.n == 104, 'callee sees its own copy');
assert(p.n == 4, 'caller struct is unchanged (pass by value)');

% A field of a type the JIT does not take (a cell) makes the struct decline,
% which must fall back cleanly rather than fail.
c = struct('c', {{1, 2}}, 'n', 3);
tot2 = 0;
for k = 1:3
    tot2 = tot2 + c.n + numel(c.c);
end
assert(tot2 == 15, 'unsupported field type falls back');

disp('SUCCESS')

function y = use(p, k)
y = p.tol * k + p.n + sum(p.v);
end

function q = build(x)
q = struct('a', x, 'b', 2*x);
end

function y = deep(r)
y = r.inner.z * r.w;
end

function p = bump(p)
p.n = p.n + 100;
end
