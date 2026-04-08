% Test logical builtins: or, and, xor, not (functional forms of |, &, ~)

% ── or: scalar ──────────────────────────────────────────────────────────
assert(or(true, false) == true);
assert(or(false, false) == false);
assert(or(true, true) == true);
assert(or(0, 0) == false);
assert(or(0, 5) == true);
assert(or(3, 0) == true);
assert(islogical(or(1, 0)));

% ── or: vector / matrix ─────────────────────────────────────────────────
v = or([1 0 1 0], [0 0 1 1]);
assert(isequal(v, logical([1 0 1 1])));
assert(islogical(v));

M = or([1 0; 0 1], [0 0; 1 1]);
assert(isequal(M, logical([1 0; 1 1])));

% ── or: scalar/tensor broadcast ─────────────────────────────────────────
assert(isequal(or(true, [0 0 0]), logical([1 1 1])));
assert(isequal(or(false, [1 0 1]), logical([1 0 1])));
assert(isequal(or([0 0 0], 1), logical([1 1 1])));

% ── or: matches | operator ──────────────────────────────────────────────
A = [1 0 1; 0 1 1];
B = [0 0 1; 1 1 0];
assert(isequal(or(A, B), A | B));

% ── and: scalar ─────────────────────────────────────────────────────────
assert(and(true, true) == true);
assert(and(true, false) == false);
assert(and(false, true) == false);
assert(and(0, 5) == false);
assert(and(3, 4) == true);
assert(islogical(and(1, 1)));

% ── and: vector / matrix ────────────────────────────────────────────────
v = and([1 0 1 1], [1 1 0 1]);
assert(isequal(v, logical([1 0 0 1])));
assert(islogical(v));

% ── and: scalar/tensor broadcast ────────────────────────────────────────
assert(isequal(and(true, [1 0 1]), logical([1 0 1])));
assert(isequal(and(false, [1 1 1]), logical([0 0 0])));

% ── and: matches & operator ─────────────────────────────────────────────
assert(isequal(and(A, B), A & B));

% ── xor: scalar ─────────────────────────────────────────────────────────
assert(xor(true, false) == true);
assert(xor(false, true) == true);
assert(xor(true, true) == false);
assert(xor(false, false) == false);
assert(xor(0, 5) == true);
assert(xor(3, 4) == false);
assert(islogical(xor(1, 0)));

% ── xor: vector ─────────────────────────────────────────────────────────
v = xor([1 0 1 1], [1 1 0 1]);
assert(isequal(v, logical([0 1 1 0])));
assert(islogical(v));

% ── not: scalar ─────────────────────────────────────────────────────────
assert(not(true) == false);
assert(not(false) == true);
assert(not(0) == true);
assert(not(5) == false);
assert(islogical(not(1)));

% ── not: vector / matrix ────────────────────────────────────────────────
v = not([1 0 1 0]);
assert(isequal(v, logical([0 1 0 1])));
assert(islogical(v));

assert(isequal(not([1 0; 0 1]), logical([0 1; 1 0])));

% ── not: matches ~ operator ─────────────────────────────────────────────
assert(isequal(not(A), ~A));

% ── Combinations used in real code (e.g. chunkie) ───────────────────────
% Pattern: or(and(cond1, cond2), cond3)
adjs = [-1 2 0; 1 -1 -1];
ich = 1;
rlself = 0.5;
chsmall = 0.1;
result = or(and(adjs(1, ich) <= 0, rlself > chsmall(1)), false);
assert(result == true);

% Pattern: or(strcmpi(s1, t), strcmpi(s2, t))
assert(or(strcmpi('a', 'A'), strcmpi('b', 'A')) == true);
assert(or(strcmpi('x', 'A'), strcmpi('y', 'A')) == false);

disp('SUCCESS');
