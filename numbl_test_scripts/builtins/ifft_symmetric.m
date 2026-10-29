% ifft(Y, 'symmetric') treats each sequence as conjugate symmetric, using
% only its first half (after padding or truncating it to n), and returns
% real data.

x = [0.3 -1.2 0.7 2.1 -0.4 0.9]';
y = fft(x);
a = ifft(y, 'symmetric');
assert(isreal(a) && max(abs(a - x)) < 1e-12);

% the second half is ignored
y2 = y;
y2(5) = 100;
y2(6) = -3i;
assert(isequal(ifft(y2, 'symmetric'), ifft(y, 'symmetric')));
assert(max(abs(ifft(y2, 'symmetric') - x)) < 1e-12);

% as the inverse of the mirrored sequence
z = [1+1i; 2-1i; 3+2i; 4; 5i];
m = [z(1); z(2); z(3); conj(z(3)); conj(z(2))];
assert(max(abs(ifft(z, 'symmetric') - real(ifft(m)))) < 1e-12);
assert(max(abs(ifft(z.', 'symmetric') - real(ifft(m.')))) < 1e-12);

% with a length and a dimension
p = [z; 0; 0; 0];
mp = [p(1:5); conj(p(4:-1:2))];
assert(max(abs(ifft(z, 8, 'symmetric') - real(ifft(mp)))) < 1e-12);
Z = [z, 2 * z];
r = ifft(Z, [], 1, 'symmetric');
assert(isequal(size(r), [5 2]) && max(max(abs(r - real(ifft([m, 2 * m]))))) < 1e-12);
assert(isequal(ifft(Z.', [], 2, 'symmetric'), r.'));
assert(isreal(ifft(Z, [], 2, 'symmetric')));

% values from MATLAB
assert(max(abs(ifft([1 2 3], 'symmetric') - [5/3 -1/3 -1/3])) < 1e-12);
assert(max(abs(ifft([1+1i 2 3 4], 'symmetric') - [2 -0.5 0 -0.5])) < 1e-12);

% single in, single out; 'nonsymmetric' is the default
s = ifft(single(y), 'symmetric');
assert(isa(s, 'single') && isreal(s));
assert(isequal(ifft(y, 'nonsymmetric'), ifft(y)));

disp('SUCCESS')
