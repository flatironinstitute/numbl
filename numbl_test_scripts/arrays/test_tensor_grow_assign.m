%% Grow column vector via 2-index range assignment (1:2:n,:)
A = [1; 2; 3];
A(1:2:5,:) = A;
assert(isequal(A, [1; 2; 2; 0; 3]));

%% Grow matrix - assign beyond current rows via scalar row index
C = [1 2; 3 4];
C(5, :) = [9 10];
assert(isequal(size(C), [5 2]));
assert(isequal(C(1,:), [1 2]));
assert(isequal(C(5,:), [9 10]));
assert(isequal(C(3,:), [0 0]));  % zero-filled gap rows

%% Grow matrix - assign beyond current columns via scalar col index
D = [1 2; 3 4];
D(:, 4) = [7; 8];
assert(isequal(size(D), [2 4]));
assert(isequal(D(:,1), [1; 3]));
assert(isequal(D(:,4), [7; 8]));
assert(isequal(D(:,3), [0; 0]));  % zero-filled gap column

%% Scalar grow via linear index
E = 1;
E(3) = 5;
assert(isequal(E, [1 0 5]));

%% Grow matrix via scalar-scalar 2-index
F = [1 2; 3 4];
F(3, 5) = 99;
assert(isequal(size(F), [3 5]));
assert(F(3, 5) == 99);
assert(F(1, 1) == 1);
assert(F(2, 2) == 4);
assert(F(3, 3) == 0);  % zero-filled

%% Grow via 1-index tensor assignment (linear indexing)
G = [10; 20; 30];
G([1 5]) = [100; 500];
assert(isequal(G, [100; 20; 30; 0; 500]));

%% Grow via 1-index tensor assignment with scalar RHS
H = [1 2 3];
H([1 6]) = 99;
assert(isequal(H, [99 2 3 0 0 99]));

disp('SUCCESS')
