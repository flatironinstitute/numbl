% The `end` keyword inside indexing: with k subscripts, `end` in the last
% position is the product of the remaining (collapsed trailing) dimensions;
% with one subscript it is numel. Covers numeric, char, logical, cell,
% string, struct and sparse bases, nested indexing, and assignment targets.

% N-D numeric with fewer subscripts than dimensions
A = reshape(1:24, 2, 3, 4);
assert(A(2, end) == 24);
assert(A(end, end) == 24);
assert(A(1, end-1) == 21);
assert(A(2, 3, end) == 24);
assert(A(1, 1, 1, end) == 1);
assert(A(end) == 24);
assert(isequal(A(1, end-1:end), [21 23]));
assert(isequal(size(A(:, end)), [2 1]));
assert(isequal(A(:, end), [23; 24]));

% Extra subscripts past ndims have extent 1
x = 1:5;
assert(x(1, end) == 5);
assert(x(end, 1) == 1);
assert(x(1, 1, end) == 1);
assert(x(end, end, end) == 5);

% Char matrix
c = ['abc'; 'def'];
assert(c(end, 1) == 'd');
assert(c(1, end) == 'c');
assert(c(end) == 'f');
assert(c(end, end) == 'f');
assert(isequal(c(end, :), 'def'));
assert(isequal(c(:, end), ['c'; 'f']));
q = 'hello';
assert(isequal(q(end:-1:1), 'olleh'));

% Logical N-D
L = false(2, 3, 2);
L(2, 3, 2) = true;
assert(L(2, end));
assert(~L(2, end-1));
assert(islogical(L(end)));

% Cell
C = {1, 2; 3, 4};
assert(C{end, 1} == 3);
assert(C{end} == 4);
assert(iscell(C(end, end)));
C{1, end+1} = 7;
assert(isequal(size(C), [2 3]));
assert(C{1, 3} == 7);

% String array
str = ["a", "b"; "c", "d"];
assert(str(end, 1) == "c");
assert(str(end) == "d");
assert(str(1, end) == "b");

% Struct array
s = struct('a', {1, 2, 3});
assert(s(end).a == 3);
s(end+1).a = 4;
assert(numel(s) == 4);
assert(s(end).a == 4);

% Sparse
S = sparse([1 0 0 0; 0 0 0 0; 0 0 0 5]);
assert(S(end, end) == 5);
assert(S(end) == 5);
assert(S(end, 1) == 0);

% Nested end resolves against the innermost base
B = [3 1 2];
M = magic(4);
assert(M(B(end), end) == M(2, 4));
assert(A(B(end), end) == 24);

% Assignment targets
y = [];
y(end+1) = 5;
y(end+1) = 6;
assert(isequal(y, [5 6]));
Z = zeros(2, 2);
Z(2, end+1) = 9;
assert(isequal(size(Z), [2 3]));
assert(Z(2, 3) == 9);
W = reshape(1:24, 2, 3, 4);
W(2, end) = 100;
assert(W(24) == 100);
assert(isequal(size(W), [2 3 4]));
cc = ['abc'; 'def'];
cc(end, end) = 'z';
assert(isequal(cc, ['abc'; 'dez']));

% Inside functions (exercises the JIT at --opt 1/2 where it applies)
assert(sum_last_col(A) == 3 * 24);
assert(sum_last_page(A) == 3 * (23 + 24));
assert(sum_end_2d(magic(4)) == 3 * 1);

disp('SUCCESS');

function r = sum_last_col(A)
  r = 0;
  for i = 1:3
    r = r + A(2, end);
  end
end

function r = sum_last_page(A)
  r = 0;
  for i = 1:3
    r = r + A(1, 3, end) + A(2, end, end);
  end
end

function r = sum_end_2d(M)
  r = 0;
  for i = 1:3
    r = r + M(end, end);
  end
end
