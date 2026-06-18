% Test nchoosek: binomial coefficient and all combinations

% Binomial coefficient, scalar inputs
assert(nchoosek(5, 4) == 5);
assert(nchoosek(10, 3) == 120);
assert(nchoosek(6, 0) == 1);
assert(nchoosek(6, 6) == 1);
assert(nchoosek(0, 0) == 1);
assert(nchoosek(52, 5) == 2598960);

% 1x1 array is a scalar -> binomial coefficient
assert(nchoosek([7], 2) == 21);

% All combinations of a numeric vector
C = nchoosek(2:2:10, 4);
expected = [2 4 6 8; 2 4 6 10; 2 4 8 10; 2 6 8 10; 4 6 8 10];
assert(isequal(C, expected));
assert(isequal(size(C), [5 4]));

% Column vector input gives the same combinations
Ccol = nchoosek((2:2:10)', 4);
assert(isequal(Ccol, expected));

% Combinations of three taken two at a time
C2 = nchoosek([10 20 30], 2);
assert(isequal(C2, [10 20; 10 30; 20 30]));

% k == numel(v) -> single row with all elements
C3 = nchoosek([1 2 3], 3);
assert(isequal(C3, [1 2 3]));

% k > numel(v) -> empty matrix with k columns
E = nchoosek([1 2 3], 5);
assert(isempty(E));
assert(isequal(size(E), [0 5]));

% k == 0 -> one empty combination (1x0)
Z = nchoosek([1 2 3], 0);
assert(isequal(size(Z), [1 0]));

% Char vector preserves char type
Cc = nchoosek('abcd', 2);
assert(ischar(Cc));
assert(isequal(Cc, ['ab'; 'ac'; 'ad'; 'bc'; 'bd'; 'cd']));

% Logical vector preserves logical type
Lc = nchoosek([true false true], 2);
assert(islogical(Lc));
assert(isequal(Lc, logical([1 0; 1 1; 0 1])));

% Complex vector
Cx = nchoosek([1+1i 2+1i 3+1i], 2);
assert(isequal(Cx, [1+1i 2+1i; 1+1i 3+1i; 2+1i 3+1i]));

% Row count equals the binomial coefficient
V = 1:6;
assert(size(nchoosek(V, 3), 1) == nchoosek(6, 3));

disp('SUCCESS');
