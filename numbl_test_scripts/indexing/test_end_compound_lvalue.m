% Test that 'end' resolves correctly in compound lvalue indexing

% end in struct field indexing: s.A(end) = value
s.A = [10, 20, 30];
s.A(end) = 99;
assert(isequal(s.A, [10, 20, 99]));

% end in nested struct indexing: s.B(end) read-back
s.B = [1, 2, 3, 4, 5];
s.B(end) = 50;
assert(isequal(s.B, [1, 2, 3, 4, 50]));

% end with colon in struct field: s.C(2:end)
s.C = [10, 20, 30, 40];
s.C(2:end) = [77, 88, 99];
assert(isequal(s.C, [10, 77, 88, 99]));

% end in cell array compound indexing
c = {[1, 2, 3], [4, 5, 6]};
c{1}(end) = 99;
assert(isequal(c{1}, [1, 2, 99]));

disp('SUCCESS');
