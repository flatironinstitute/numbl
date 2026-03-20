% Test copy-on-write semantics for zero-copy reshape.
% Ensures that shared buffers are correctly copied before mutation.

%% Basic: modify source after reshape
a = [1 2 3 4 5 6];
b = reshape(a, 2, 3);
a(1) = 99;
assert(a(1) == 99, 'source should be modified');
assert(b(1,1) == 1, 'reshaped copy should be unaffected by source mutation');

%% Basic: modify reshaped after reshape
a = [1 2 3 4 5 6];
b = reshape(a, 2, 3);
b(1,1) = 77;
assert(a(1) == 1, 'source should be unaffected by reshaped mutation');
assert(b(1,1) == 77, 'reshaped should be modified');

%% Chain of reshapes: a -> b -> c, mutate each independently
a = [1 2 3 4 5 6 7 8 9 10 11 12];
b = reshape(a, 3, 4);
c = reshape(b, 4, 3);
a(1) = 100;
b(1,1) = 200;
c(1,1) = 300;
assert(a(1) == 100, 'a should have its own value');
assert(b(1,1) == 200, 'b should have its own value');
assert(c(1,1) == 300, 'c should have its own value');
% Check the rest of the data is intact
assert(a(2) == 2, 'a rest intact');
assert(b(2,1) == 2, 'b rest intact');
assert(c(2,1) == 2, 'c rest intact');

%% Multiple reshapes from same source
a = [10 20 30 40 50 60];
b = reshape(a, 2, 3);
c = reshape(a, 3, 2);
d = reshape(a, 6, 1);
a(1) = 999;
assert(b(1,1) == 10, 'b unaffected');
assert(c(1,1) == 10, 'c unaffected');
assert(d(1) == 10, 'd unaffected');
b(1,1) = 888;
assert(c(1,1) == 10, 'c unaffected by b mutation');
assert(d(1) == 10, 'd unaffected by b mutation');

%% Reshape in a loop (common pattern in chebfun)
a = [1 2 3 4 5 6 7 8 9];
results = zeros(1, 3);
for i = 1:3
    b = reshape(a, 3, 3);
    results(i) = b(1,1);
    a(1) = a(1) + 10;
end
assert(results(1) == 1, 'iter 1 should see original');
assert(results(2) == 11, 'iter 2 should see modified');
assert(results(3) == 21, 'iter 3 should see modified again');

%% Reshape then grow (horzcat extends the array)
a = [1 2 3 4];
b = reshape(a, 2, 2);
a = [a 5 6];  % a gets a new buffer
assert(length(a) == 6, 'a extended');
assert(all(b(:)' == [1 2 3 4]), 'b unaffected by a extension');

%% Reshape complex array
a = [1+2i 3+4i 5+6i 7+8i];
b = reshape(a, 2, 2);
a(1) = 99+99i;
assert(real(b(1,1)) == 1, 'complex reshape: b real unaffected');
assert(imag(b(1,1)) == 2, 'complex reshape: b imag unaffected');
assert(real(a(1)) == 99, 'complex reshape: a real modified');
b(2,2) = 0;
assert(real(a(4)) == 7, 'complex reshape: a unaffected by b mutation');

%% Reshape of a slice (slice already copies, so no double-copy needed)
a = [1 2 3 4 5 6 7 8 9 10 11 12];
a = reshape(a, 3, 4);
row = a(1,:);       % extracts [1 4 7 10]
b = reshape(row, 2, 2);
row(1) = 999;
assert(b(1,1) == 1, 'reshape of slice: b unaffected by row mutation');
assert(a(1,1) == 1, 'original a unaffected');

%% Reshape then use as function argument (pass by value semantics)
a = [1 2 3 4 5 6];
b = reshape(a, 2, 3);
c = modify_and_return(b);
assert(b(1,1) == 1, 'b should be unaffected after function call');
assert(c(1,1) == 42, 'c should have modified value');

%% 3D reshape
a = 1:24;
b = reshape(a, 2, 3, 4);
c = reshape(b, 4, 6);
a(1) = 999;
assert(b(1,1,1) == 1, '3D reshape: b unaffected');
assert(c(1,1) == 1, '3D reshape: c unaffected');
b(1,1,1) = 888;
assert(c(1,1) == 1, '3D reshape chain: c unaffected by b');

%% Reshape then indexed assignment that grows
a = [1 2 3 4];
b = reshape(a, 2, 2);
b(3,3) = 9;  % grows b to 3x3
assert(all(a == [1 2 3 4]), 'a unaffected by b growth');
assert(b(3,3) == 9, 'b grew correctly');
assert(b(1,1) == 1, 'b preserved original data');

%% Reshape preserves values correctly (not just COW)
a = sin(1:1000);
b = reshape(a, 10, 100);
c = reshape(b, 50, 20);
assert(abs(a(1) - b(1,1)) < 1e-15, 'values match after reshape');
assert(abs(a(1) - c(1,1)) < 1e-15, 'values match after double reshape');
assert(abs(a(11) - b(1,2)) < 1e-15, 'col-major order preserved');

disp('SUCCESS');


function out = modify_and_return(x)
    x(1,1) = 42;
    out = x;
end
