% Assigning a struct into a struct array element requires the same set of
% fields (order may differ). Adding a field via s(k).f = v is allowed and
% adds the field to every element, without affecting copies.

% superset of fields: error, and copy unchanged
s(1).a = 1; s(2).a = 2;
s2 = s;
threw = false;
try
  s(3) = struct('a', 3, 'b', 4);
catch err
  threw = true;
  assert(strcmp(err.message, 'Subscripted assignment between dissimilar structures.'));
end
assert(threw);
assert(isequal(fieldnames(s2(1)), {'a'}));
assert(isequal(fieldnames(s), {'a'}));
assert(numel(s) == 2);

% subset of fields: error
t(1).a = 1; t(1).b = 2; t(2).a = 3; t(2).b = 4;
threw = false;
try
  t(3) = struct('a', 5);
catch
  threw = true;
end
assert(threw);
assert(numel(t) == 2);

% scalar struct target: error on different field set
y.a = 1;
threw = false;
try
  y(2) = struct('a', 2, 'b', 3);
catch
  threw = true;
end
assert(threw);
threw = false;
try
  y(1) = struct('b', 2);
catch
  threw = true;
end
assert(threw);

% empty struct array with fields: error
z = struct('a', {});
threw = false;
try
  z(1) = struct('b', 1);
catch
  threw = true;
end
assert(threw);

% nested target
p.q(1).a = 1;
threw = false;
try
  p.q(2) = struct('a', 2, 'b', 3);
catch
  threw = true;
end
assert(threw);

% same fields in a different order: allowed, target order kept
u(1).a = 1; u(1).b = 2;
u(2) = struct('b', 5, 'a', 6);
assert(isequal(fieldnames(u), {'a'; 'b'}));
assert(isequal(fieldnames(u(2)), {'a'; 'b'}));
assert(u(2).a == 6 && u(2).b == 5);

% matching fields: allowed, copy unaffected
x(1).a = 1;
x2 = x;
x(2) = struct('a', 2);
assert(numel(x) == 2);
assert(numel(x2) == 1);

% field added through a new element: all elements gain it, copy does not
v(1).a = 1; v(2).a = 2;
v2 = v;
v(3).b = 4;
assert(isequal(fieldnames(v), {'a'; 'b'}));
assert(isequal(fieldnames(v(1)), {'a'; 'b'}));
assert(isequal(fieldnames(v(3)), {'a'; 'b'}));
assert(isempty(v(1).b) && isempty(v(3).a));
assert(v(3).b == 4);
assert(isequal(fieldnames(v2), {'a'}));
assert(isequal(fieldnames(v2(1)), {'a'}));
assert(isequal(fieldnames(v2(2)), {'a'}));

% field added through an existing element: copy unaffected
w(1).a = 1; w(2).a = 2;
w2 = w;
w(1).b = 5;
assert(isequal(fieldnames(w(2)), {'a'; 'b'}));
assert(isempty(w(2).b));
assert(isequal(fieldnames(w2(1)), {'a'}));
assert(isequal(fieldnames(w2(2)), {'a'}));

% field added on an empty struct array
e = struct('a', {});
e(1).b = 2;
assert(isequal(fieldnames(e), {'a'; 'b'}));

% field added through a struct array held in a struct field
h.items(1).a = 1; h.items(2).a = 2;
h2 = h;
h.items(2).c = 3;
assert(isequal(fieldnames(h2.items(1)), {'a'}));
assert(isequal(fieldnames(h.items(1)), {'a'; 'c'}));

disp('SUCCESS')
