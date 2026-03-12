% Test multi-level inheritance: BaseShape2_ -> RectShape2_ -> SquareShape2_
% Covers: inherited properties, overridden methods, super constructors, isa

% --- Base class ---
b = BaseShape2_('red');
assert(strcmp(__inferred_type_str(b), 'ClassInstance<BaseShape2_>'));
assert(strcmp(b.Color, 'red'));
assert(b.describe() == 0, 'base describe');
assert(b.color_code() == 1, 'red = 1');

% --- Middle class (inherits Color, overrides describe) ---
r = RectShape2_(3, 4, 'blue');
assert(strcmp(__inferred_type_str(r), 'ClassInstance<RectShape2_>'));
assert(r.Width == 3);
assert(r.Height == 4);
assert(strcmp(r.Color, 'blue'), 'inherited Color from super constructor');
assert(r.area() == 12, 'rect area');
assert(r.describe() == 1, 'rect overrides describe');
assert(r.color_code() == 2, 'inherited color_code, blue = 2');

% Default color
r2 = RectShape2_(5, 6);
assert(strcmp(__inferred_type_str(r2), 'ClassInstance<RectShape2_>'));
assert(strcmp(r2.Color, 'black'));

% --- Leaf class (inherits Color, Width, Height, area; overrides describe) ---
s = SquareShape2_(7, 'red');
assert(strcmp(__inferred_type_str(s), 'ClassInstance<SquareShape2_>'));
assert(s.Width == 7);
assert(s.Height == 7);
assert(strcmp(s.Color, 'red'));
assert(s.area() == 49, 'inherited area from rect');
assert(s.describe() == 2, 'square overrides describe');
assert(s.color_code() == 1, 'inherited color_code from base');
assert(abs(s.diagonal() - 7 * sqrt(2)) < 1e-10);

% --- isa checks across hierarchy ---
assert(isa(b, 'BaseShape2_'));
assert(~isa(b, 'RectShape2_'));
assert(~isa(b, 'SquareShape2_'));

assert(isa(r, 'BaseShape2_'), 'rect is-a base');
assert(isa(r, 'RectShape2_'));
assert(~isa(r, 'SquareShape2_'));

assert(isa(s, 'BaseShape2_'), 'square is-a base (transitive)');
assert(isa(s, 'RectShape2_'), 'square is-a rect');
assert(isa(s, 'SquareShape2_'));

% --- Dispatch with inherited methods ---
% color_code is defined only on BaseShape2_, inherited down
assert(color_code(s) == 1, 'func-call dispatch to inherited method');
assert(color_code(r) == 2);
assert(color_code(b) == 1);

% describe is overridden at each level
assert(describe(b) == 0);
assert(describe(r) == 1);
assert(describe(s) == 2);

disp('SUCCESS')
