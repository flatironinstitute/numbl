% Stage 12 — scalar struct field read inside a JIT loop. Mirrors the
% chunkie pattern `chnkr.k`, `chnkr.nch`, `opts.rho`, `opts.tol` where
% a small options/parameters struct is built outside a tight loop and
% its scalar fields are read many times inside.
%
% Lowering: `lowerExpr` recognizes `Member(Ident(s), f)` where `s` has
% a struct type with a known scalar numeric field `f` and emits a
% `MemberRead` IR node. Codegen walks the body, collects every
% `(baseName, fieldName)` pair, and hoists
% `var $s_f = s.fields.get("f")` at function entry. Each per-iter read
% then becomes a bare local load.
%
% `assert_jit_compiled()` is placed inside every outer loop body to
% assert the surrounding loop got JIT-compiled — if the marker call
% survives to the interpreter (because lowering bailed), it throws.

% 1) Basic: read a single scalar field many times, verify arithmetic.
opts.k = 16;
opts.tol = 1e-6;
opts.rho = 1.8;
total = 0;
for i = 1:100
    assert_jit_compiled();
    total = total + opts.k * i;
end
% sum(i=1..100) = 5050; * 16 = 80800
assert(total == 80800, '1: basic sum');

% 2) Multiple fields per iter, with an if/else condition using a field.
acc1 = 0;
acc2 = 0;
for i = 1:50
    assert_jit_compiled();
    v = opts.k * i + opts.rho;
    if v > opts.tol
        acc1 = acc1 + v;
    else
        acc2 = acc2 + v;
    end
end
% opts.k * i + opts.rho is always > 1e-6 for i>=1, so acc2 stays 0.
assert(acc2 == 0, '2: acc2 is zero');
% acc1 = sum(i=1..50) * 16 + 50 * 1.8 = 20400 + 90 = 20490
assert(abs(acc1 - 20490) < 1e-9, '2: acc1 value');

% 3) Field in a loop bound — exercises struct read outside inner body
%    but still inside the JIT loop. (Uses the field in the driver.)
opts2.n = 30;
opts2.scale = 2.5;
sum3 = 0;
for i = 1:opts2.n
    assert_jit_compiled();
    sum3 = sum3 + i * opts2.scale;
end
% sum(i=1..30) = 465; * 2.5 = 1162.5
assert(abs(sum3 - 1162.5) < 1e-9, '3: loop bound from field');

% 4) Boolean-typed field — ensure non-number numeric scalars work too.
cfg.active = true;
cfg.step = 3;
cnt = 0;
for i = 1:20
    assert_jit_compiled();
    if cfg.active
        cnt = cnt + cfg.step;
    end
end
% cnt = 20 * 3 = 60
assert(cnt == 60, '4: boolean field gate');

% 5) Nested loops with struct reads at both levels.
params.a = 5;
params.b = 2;
grand = 0;
for i = 1:10
    assert_jit_compiled();
    for j = 1:5
        grand = grand + params.a * i + params.b * j;
    end
end
% = sum_i sum_j (5*i + 2*j)
% = 5 * sum_i(i) * 5 + 2 * sum_j(j) * 10
% = 5 * 55 * 5 + 2 * 15 * 10 = 1375 + 300 = 1675
assert(grand == 1675, '5: nested loop struct reads');

% 6) Reading the same field twice per iter (hoist must not duplicate).
consts.pi_approx = 3.14159;
s6 = 0;
for i = 1:10
    assert_jit_compiled();
    s6 = s6 + consts.pi_approx + consts.pi_approx * i;
end
% = 10 * pi + pi * sum(1..10) = 10 * 3.14159 + 3.14159 * 55 = 65 * 3.14159
assert(abs(s6 - 65 * 3.14159) < 1e-9, '6: same field twice');

disp('SUCCESS');
