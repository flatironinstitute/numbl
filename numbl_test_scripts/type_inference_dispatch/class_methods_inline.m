% Test methods defined inline: chained calls, method calling method,
% method returning new instances, multiple return values, method using
% result of another method as constructor arg

v1 = Vector2D_(3, 4);
assert(strcmp(__inferred_type_str(v1), 'ClassInstance<Vector2D_>'));

% --- Basic ---
assert(v1.magnitude() == 5);

v2 = Vector2D_(1, 2);
v3 = v1.add(v2);
assert(strcmp(__inferred_type_str(v3), 'ClassInstance<Vector2D_>'));
assert(v3.X == 4);
assert(v3.Y == 6);

% --- Chained method calls: scale returns Vector2D_, call magnitude on it ---
assert(strcmp(__inferred_type_str(v1.scale(2)), 'ClassInstance<Vector2D_>'));
assert(v1.scale(2).magnitude() == 10, 'chain: scale then magnitude');

% --- Three-deep chain: add -> scale -> magnitude ---
r = v1.add(Vector2D_(0, 0)).scale(3).magnitude();
assert(r == 15, 'chain: add->scale->magnitude');

% --- Method that calls multiple other methods internally ---
v4 = Vector2D_(3, 4);
v5 = Vector2D_(1, 0);
proj = v4.project_onto(v5);
assert(proj.X == 3);
assert(proj.Y == 0);

% --- Projection onto diagonal ---
v6 = Vector2D_(4, 0);
v7 = Vector2D_(1, 1);
proj2 = v6.project_onto(v7);
assert(abs(proj2.X - 2) < 1e-10);
assert(abs(proj2.Y - 2) < 1e-10);

% --- Chain: normalized calls magnitude and scale ---
n = Vector2D_(3, 4).normalized();
assert(abs(n.magnitude() - 1) < 1e-10);
assert(abs(n.X - 0.6) < 1e-10);

% --- Chained normalized then dot_prod ---
d = Vector2D_(3, 4).normalized().dot_prod(Vector2D_(0, 5).normalized());
assert(abs(d - 0.8) < 1e-10, 'dot of normalized vectors');

% --- Multiple return values from a method ---
[mag, angle] = v1.polar();
assert(mag == 5);
assert(abs(angle - atan2(4, 3)) < 1e-10);

% --- Constructor result used directly in chained call ---
assert(Vector2D_(6, 8).magnitude() == 10);
assert(Vector2D_(1, 0).add(Vector2D_(0, 1)).magnitude() == sqrt(2));

% --- Method result as constructor argument ---
v10 = Vector2D_(v1.magnitude(), v2.magnitude());
assert(strcmp(__inferred_type_str(v10), 'ClassInstance<Vector2D_>'));
assert(v10.X == 5);
assert(abs(v10.Y - sqrt(5)) < 1e-10);

% --- Accumulate via methods in a loop ---
acc = Vector2D_(0, 0);
for i = 1:4
    assert(strcmp(__inferred_type_str(i), "Number"));
    acc = acc.add(Vector2D_(i, i * 2));
end
assert(strcmp(__inferred_type_str(acc), 'ClassInstance<Vector2D_>'));
assert(acc.X == 10);
assert(acc.Y == 20);

% --- Call method on result of local helper function ---
v_from_helper = make_vec(5, 12);
assert(strcmp(__inferred_type_str(v_from_helper), 'ClassInstance<Vector2D_>'));
assert(v_from_helper.magnitude() == 13);

% --- Chained call on result of local helper ---
assert(make_vec(3, 4).scale(2).magnitude() == 10);

disp('SUCCESS')

function r = make_vec(x, y)
    r = Vector2D_(x, y);
end
