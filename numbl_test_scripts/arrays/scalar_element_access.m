% Test scalar element access for 1D, 2D, 3D, and cell arrays.
% Exercises the fast paths in runtimeIndexing.ts.

% ── 1D linear indexing ──
x = [10 20 30 40 50];
assert(x(1) == 10);
assert(x(3) == 30);
assert(x(5) == 50);

% 1D in a loop
s = 0;
for i = 1:5
    s = s + x(i);
end
assert(s == 150);

% ── 2D indexing (col-major) ──
A = [1 2 3; 4 5 6; 7 8 9];
assert(A(1,1) == 1);
assert(A(2,1) == 4);
assert(A(3,1) == 7);
assert(A(1,2) == 2);
assert(A(2,3) == 6);
assert(A(3,3) == 9);

% 2D in nested loop
s = 0;
for i = 1:3
    for j = 1:3
        s = s + A(i,j);
    end
end
assert(s == 45);

% 2D linear indexing (column-major order)
assert(A(1) == 1);
assert(A(2) == 4);
assert(A(4) == 2);
assert(A(9) == 9);

% ── 3D indexing ──
B = zeros(2, 3, 4);
for i = 1:2
    for j = 1:3
        for k = 1:4
            B(i,j,k) = i * 100 + j * 10 + k;
        end
    end
end
assert(B(1,1,1) == 111);
assert(B(2,1,1) == 211);
assert(B(1,3,1) == 131);
assert(B(2,3,4) == 234);
assert(B(1,2,3) == 123);

% 3D in nested loop — sum all elements
s = 0;
for i = 1:2
    for j = 1:3
        for k = 1:4
            s = s + B(i,j,k);
        end
    end
end
expected = 0;
for i = 1:2
    for j = 1:3
        for k = 1:4
            expected = expected + i * 100 + j * 10 + k;
        end
    end
end
assert(s == expected);

% ── 4D indexing ──
C = zeros(2, 2, 2, 2);
C(1,1,1,1) = 1;
C(2,1,1,1) = 2;
C(1,2,1,1) = 3;
C(1,1,2,1) = 4;
C(1,1,1,2) = 5;
C(2,2,2,2) = 6;
assert(C(1,1,1,1) == 1);
assert(C(2,1,1,1) == 2);
assert(C(1,2,1,1) == 3);
assert(C(1,1,2,1) == 4);
assert(C(1,1,1,2) == 5);
assert(C(2,2,2,2) == 6);

% ── Cell array scalar access ──
c = {10, 'hello', [1 2 3]};
assert(c{1} == 10);
assert(strcmp(c{2}, 'hello'));
v = c{3};
assert(v(1) == 1);
assert(v(2) == 2);
assert(v(3) == 3);

% Cell access in loop
vals = {1, 2, 3, 4, 5};
s = 0;
for i = 1:5
    s = s + vals{i};
end
assert(s == 15);

% ── Complex tensor access ──
Z = [1+2i, 3+4i; 5+6i, 7+8i];
assert(Z(1,1) == 1+2i);
assert(Z(2,1) == 5+6i);
assert(Z(1,2) == 3+4i);
assert(Z(2,2) == 7+8i);

% Complex 1D access
assert(Z(1) == 1+2i);
assert(Z(2) == 5+6i);
assert(Z(3) == 3+4i);

% ── Bounds checking ──
caught = false;
try
    x(0);
catch
    caught = true;
end
assert(caught);

caught = false;
try
    x(6);
catch
    caught = true;
end
assert(caught);

caught = false;
try
    A(4, 1);
catch
    caught = true;
end
assert(caught);

caught = false;
try
    A(1, 4);
catch
    caught = true;
end
assert(caught);

disp('SUCCESS');
