% Test magic square generation

% Test odd order (3)
M3 = magic(3);
assert(isequal(size(M3), [3 3]));
assert(sum(M3(1,:)) == 15);
assert(sum(M3(2,:)) == 15);
assert(sum(M3(3,:)) == 15);
assert(sum(M3(:,1)) == 15);
assert(sum(M3(:,2)) == 15);
assert(sum(M3(:,3)) == 15);

% Test odd order (5)
M5 = magic(5);
assert(isequal(size(M5), [5 5]));
s5 = 5 * (5^2 + 1) / 2;
for i = 1:5
    assert(sum(M5(i,:)) == s5);
    assert(sum(M5(:,i)) == s5);
end

% Test doubly even order (4)
M4 = magic(4);
assert(isequal(size(M4), [4 4]));
s4 = 4 * (16 + 1) / 2;
for i = 1:4
    assert(sum(M4(i,:)) == s4);
    assert(sum(M4(:,i)) == s4);
end

% Test doubly even order (8)
M8 = magic(8);
assert(isequal(size(M8), [8 8]));
s8 = 8 * (64 + 1) / 2;
for i = 1:8
    assert(sum(M8(i,:)) == s8);
    assert(sum(M8(:,i)) == s8);
end

% Test singly even order (6)
M6 = magic(6);
assert(isequal(size(M6), [6 6]));
s6 = 6 * (36 + 1) / 2;
for i = 1:6
    assert(sum(M6(i,:)) == s6);
    assert(sum(M6(:,i)) == s6);
end

% Test that magic(10) returns a 10x10 matrix (singly even path)
M10 = magic(10);
assert(isequal(size(M10), [10 10]));

disp('SUCCESS');
