% Test that space-separated imaginary literals in matrix expressions
% are parsed as separate elements, not binary subtraction.
% e.g. [0 -2i] should be two elements, not 0-2i.

% Basic case: space before minus means separate element
A = [0 -2i];
assert(numel(A) == 2, '[0 -2i] should have 2 elements');
assert(A(1) == 0, 'first element should be 0');
assert(A(2) == -2i, 'second element should be -2i');

% Same with plus
B = [0 +2i];
assert(numel(B) == 2, '[0 +2i] should have 2 elements');
assert(B(2) == 2i, 'second element should be +2i');

% No space means binary subtraction
C = [0-2i];
assert(numel(C) == 1, '[0-2i] should have 1 element');
assert(C == -2i, '0-2i should equal -2i');

% Mixed real and imaginary in rows
D = [1 0 -0.5i 0; 0 -1.5i 0 0.25i];
assert(all(size(D) == [2 4]), 'D should be 2x4');
assert(D(1,1) == 1);
assert(D(1,3) == -0.5i);
assert(D(2,2) == -1.5i);
assert(D(2,4) == 0.25i);

% Float with imaginary suffix
E = [1 -1.237209415222620i];
assert(numel(E) == 2, 'float imaginary should be separate element');
assert(E(1) == 1);
assert(abs(E(2) - (-1.237209415222620i)) < 1e-15);

% Multiple imaginary elements in a row
F = [1 -2i 3 +4i -5i];
assert(numel(F) == 5, 'F should have 5 elements');
assert(F(1) == 1);
assert(F(2) == -2i);
assert(F(3) == 3);
assert(F(4) == 4i);
assert(F(5) == -5i);

% Binary subtraction with no space still works
G = [1+2i 3+4i];
assert(numel(G) == 2);
assert(G(1) == 1+2i);
assert(G(2) == 3+4i);

% j variant
H = [0 -3j];
assert(numel(H) == 2, '[0 -3j] should have 2 elements');
assert(H(2) == -3i);

disp('SUCCESS');
