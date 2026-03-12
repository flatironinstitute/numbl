% Test that 'end' keyword works correctly in dot-indexing expressions
% e.g., s.field([1, end]) should resolve 'end' to the size of s.field

% Struct field indexing with end
s = struct('domain', [10, 20, 30, 40]);
result = s.domain([1, end]);
assert(result(1) == 10);
assert(result(2) == 40);

% end in arithmetic inside dot indexing
result2 = s.domain(end-1);
assert(result2 == 30);

% Nested struct
s2 = struct('data', struct('values', [5, 6, 7, 8, 9]));
result3 = s2.data.values([2, end]);
assert(result3(1) == 6);
assert(result3(2) == 9);

disp('SUCCESS');
