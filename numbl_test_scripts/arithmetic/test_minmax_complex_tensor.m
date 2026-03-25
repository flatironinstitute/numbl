% Test min/max with complex arguments (scalar, tensor, and mixed)

% Two complex scalars
r1 = min(3+4i, 1+1i);
assert(r1 == 1+1i, 'min of two complex scalars by magnitude');

r2 = max(3+4i, 1+1i);
assert(r2 == 3+4i, 'max of two complex scalars by magnitude');

% Complex scalar vs real scalar
r3 = min(10, 2+1i);
assert(r3 == 2+1i);

r4 = max(10, 2+1i);
assert(r4 == 10);

% Complex tensor reduction (1-arg)
v = [3+4i, 1+1i, 2+0i];
r5 = min(v);
assert(r5 == 1+1i);

r6 = max(v);
assert(r6 == 3+4i);

% Two complex tensors element-wise (2-arg)
a = [1+2i, 5+0i];
b = [3+0i, 2+3i];
r7 = min(a, b);
% |1+2i|~2.24 vs |3|=3 -> 1+2i; |5|=5 vs |2+3i|~3.6 -> 2+3i
assert(r7(1) == 1+2i);
assert(r7(2) == 2+3i);

r8 = max(a, b);
assert(r8(1) == 3);
assert(r8(2) == 5);

% Real tensor vs complex scalar (2-arg, broadcasting)
t = [1, 5, 3];
c = 2 + 1i;
r9 = min(t, c);
% |2+1i|~2.236: min(1, 2+1i)->1, min(5, 2+1i)->2+1i, min(3, 2+1i)->2+1i
assert(r9(1) == 1);
assert(r9(2) == 2+1i);
assert(r9(3) == 2+1i);
assert(~isreal(r9), 'min(realTensor, complexScalar) should be complex');

r10 = max(t, c);
assert(r10(1) == 2+1i);
assert(r10(2) == 5);
assert(r10(3) == 3);

disp('SUCCESS');
