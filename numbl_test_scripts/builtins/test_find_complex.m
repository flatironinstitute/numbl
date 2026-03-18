% Test find with complex dense tensors preserves imaginary parts

%% find on complex dense matrix, 3 outputs
A = [1+2i 0; 0 3-4i];
[r, c, v] = find(A);
assert(isequal(r, [1; 2]));
assert(isequal(c, [1; 2]));
assert(isequal(v, [1+2i; 3-4i]));

%% find on complex vector
B = [0 2+3i 0 1i];
[r2, c2, v2] = find(B);
assert(isequal(v2, [2+3i; 1i]));

disp('SUCCESS')
