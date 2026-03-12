% Test class methods in separate @Class files: hard scenarios
% Separate methods calling other separate methods, calling classdef methods,
% chained calls on results, deep chains

a = SepVec(1, 0, 0);
b = SepVec(0, 1, 0);
c = SepVec(2, 3, 4);
d = SepVec(1, 1, 1);
assert(strcmp(__inferred_type_str(a), 'ClassInstance<SepVec>'));
assert(strcmp(__inferred_type_str(d), 'ClassInstance<SepVec>'));

% --- Basic separate-file method ---
assert(a.dot_product(b) == 0);
assert(c.dot_product(d) == 9);

% --- Separate method returning new instance, then classdef method ---
assert(abs(a.add_vec(b).magnitude() - sqrt(2)) < 1e-10);

% --- Chain: separate -> separate (cross_product then dot_product) ---
cp = a.cross_product(b);
assert(strcmp(__inferred_type_str(cp), 'ClassInstance<SepVec>'));
assert(cp.dot_product(a) == 0, 'cross product orthogonal to input a');
assert(cp.dot_product(b) == 0, 'cross product orthogonal to input b');

% --- Separate method calling classdef method internally (normalized) ---
n = c.normalized();
assert(abs(n.magnitude() - 1) < 1e-10, 'normalized via separate file');

% --- Separate method calling other separate methods internally (angle_between) ---
ang = a.angle_between(b);
assert(abs(ang - pi/2) < 1e-10, 'orthogonal angle = pi/2');

v1 = SepVec(1, 1, 0);
v2 = SepVec(1, 0, 0);
assert(abs(v1.angle_between(v2) - pi/4) < 1e-10, 'pi/4 angle');

% --- Deep chain: cross_product -> normalized -> magnitude ---
n2 = SepVec(1, 0, 0).cross_product(SepVec(0, 1, 0)).normalized();
assert(abs(n2.magnitude() - 1) < 1e-10);

% --- Chained add_vec in a loop ---
acc = SepVec(0, 0, 0);
for i = 1:3
    assert(strcmp(__inferred_type_str(i), "Number"));
    acc = acc.add_vec(SepVec(i, i*2, i*3));
end
assert(strcmp(__inferred_type_str(acc), 'ClassInstance<SepVec>'));
assert(acc.X == 6);
assert(acc.Y == 12);
assert(acc.Z == 18);

% --- Three-step chain: all separate-file methods ---
r = a.cross_product(b).add_vec(SepVec(1, 1, 0)).dot_product(SepVec(1, 1, 1));
% cross(x,y)=(0,0,1), add(1,1,0)=(1,1,1), dot(1,1,1)=3
assert(r == 3);

% --- Scale (classdef) then dot_product (separate) ---
assert(c.scale(2).dot_product(d) == 18);

% --- Normalized then cross_product (separate calling separate chain) ---
nx = SepVec(3, 0, 0).normalized();
ny = SepVec(0, 5, 0).normalized();
cp2 = nx.cross_product(ny);
assert(abs(cp2.Z - 1) < 1e-10);

disp('SUCCESS')
