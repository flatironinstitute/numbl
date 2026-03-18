% Test that sum on a sparse row vector returns a scalar, not the row itself
v = sparse([1 2 3]);
s = sum(v);
assert(isequal(s, 6));
assert(isscalar(s));

% Also check that explicit dim=2 works the same way
s2 = sum(v, 2);
assert(isequal(s2, 6));

% dim=1 on a row vector should return the same row vector
s1 = sum(v, 1);
assert(isequal(s1, sparse([1 2 3])));

% Column vector: default sum should reduce along dim 1 to a scalar
vc = sparse([1; 2; 3]);
sc = sum(vc);
assert(isequal(sc, sparse(6)));

disp('SUCCESS')
