% Auto-grow on indexed assignment to a non-existent struct field.
% MATLAB: `s.a(3) = 5` initializes `s.a` as `[0 0 5]`. The field's
% default depends on the operation that follows: indexing → empty
% tensor (auto-grow); member access → empty struct.

% ── 1. Variable doesn't exist; bare-variable indexed assign ─────────
clear u
u(3) = 5;
assert(length(u) == 3, sprintf('1: length(u) expected 3, got %d', length(u)));
assert(u(1) == 0 && u(2) == 0 && u(3) == 5, '1: u expected [0 0 5]');

% ── 2. Existing empty struct, field doesn't exist ───────────────────
s = struct();
s.a(3) = 5;
assert(length(s.a) == 3, sprintf('2: length(s.a) expected 3, got %d', length(s.a)));
assert(s.a(1) == 0 && s.a(2) == 0 && s.a(3) == 5, '2: s.a expected [0 0 5]');

% ── 3. Variable doesn't exist; field-indexed assign auto-creates ────
clear s
s.a(3) = 5;
assert(length(s.a) == 3, sprintf('3: length(s.a) expected 3, got %d', length(s.a)));
assert(s.a(1) == 0 && s.a(2) == 0 && s.a(3) == 5, '3: s.a expected [0 0 5]');

% ── 4. Existing field set to []; index assign grows it ──────────────
s = struct();
s.a = [];
s.a(3) = 5;
assert(length(s.a) == 3, '4: length(s.a) expected 3');
assert(s.a(3) == 5, '4: s.a(3) == 5');

% ── 5. Member chain: nested struct field auto-create still struct ──
clear s
s.a.b = 7;
assert(s.a.b == 7, '5: s.a.b == 7');

% ── 6. Member then index: nested chain creates struct then tensor ──
clear s
s.a.b(3) = 9;
assert(length(s.a.b) == 3, '6: length(s.a.b) expected 3');
assert(s.a.b(3) == 9 && s.a.b(1) == 0, '6: s.a.b expected [0 0 9]');

% ── 7. Cell auto-grow on struct field ──────────────────────────────
clear s
s.a{3} = 'hello';
assert(length(s.a) == 3, '7: length(s.a) expected 3');
assert(strcmp(s.a{3}, 'hello'), '7: s.a{3} expected ''hello''');

% ── 8. 2-D auto-grow on struct field ───────────────────────────────
clear s
s.a(2, 3) = 5;
assert(size(s.a, 1) == 2 && size(s.a, 2) == 3, '8: size(s.a) expected [2 3]');
assert(s.a(2, 3) == 5, '8: s.a(2,3) == 5');

% ── 9. Multiple field auto-creates on the same struct ──────────────
clear s
s.x(3) = 5;
s.y(2) = 7;
assert(s.x(3) == 5 && length(s.x) == 3, '9: s.x ok');
assert(s.y(2) == 7 && length(s.y) == 2, '9: s.y ok');

% ── 10. After auto-grow, COW still applies on shared struct ─────────
clear s
s.a(3) = 5;
t = s;
t.a(1) = 99;
assert(s.a(1) == 0, '10: s.a(1) unchanged');
assert(t.a(1) == 99, '10: t.a(1) mutated');
assert(s.a(3) == 5 && t.a(3) == 5, '10: s.a(3) and t.a(3) intact');

disp('SUCCESS')
