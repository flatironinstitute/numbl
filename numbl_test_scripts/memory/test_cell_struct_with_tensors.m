% Cells and structs hold tensors.

% Cell holding tensors.
c = { [1, 2, 3], [4, 5, 6], [7, 8, 9] };
assert(isequal(c{1}, [1, 2, 3]));

% Reassign a cell entry — the old tensor's refcount must drop.
c{2} = [40, 50, 60];
assert(isequal(c{2}, [40, 50, 60]));
assert(isequal(c{1}, [1, 2, 3]) && isequal(c{3}, [7, 8, 9]));

% Mutate a tensor inside a cell via subsasgn.
c{1}(2) = 200;
assert(isequal(c{1}, [1, 200, 3]));

% Pass cell into a function and back.
d = make_cell_with_tensors();
assert(isequal(d{1}, (1:5)) && isequal(d{2}, (10:10:50)));
total = 0;
for i = 1:length(d)
  total = total + sum(d{i});
end
assert(total == sum(1:5) + sum(10:10:50) + 3);

% Struct with tensor fields.
s.a = [1, 2, 3];
s.b = [4, 5, 6];
s.a(2) = 99;
assert(isequal(s.a, [1, 99, 3]));
assert(isequal(s.b, [4, 5, 6]));

% Reassign struct field — old tensor released.
s.a = [100, 200];
assert(isequal(s.a, [100, 200]));

% Function consuming and returning a struct.
out = bump_field(s);
assert(out.a(1) == 101, 'returned struct field should reflect mutation');
assert(s.a(1) == 100, 'caller struct unchanged');

disp('SUCCESS')

function c = make_cell_with_tensors()
  c = { 1:5, 10:10:50, ones(1, 3) };
end

function s = bump_field(s)
  s.a(1) = s.a(1) + 1;
end
