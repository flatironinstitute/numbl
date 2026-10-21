% Mixed real/complex * and / follow MATLAB: a complex operand with an
% all-zero imaginary part acts as real, and a real operand is not promoted
% to complex, so NaN or Inf in it does not make NaN imaginary parts.

x = complex([NaN; 1; Inf], [0; 0; 0]); r = [2; 3; 4]; M = [1 2; 3 4; 5 6];
assert(isequal(imag(x.*r), [0; 0; 0]) && isreal(x.*r));
assert(isequal(imag(r.*x), [0; 0; 0]));
assert(isequal(imag(x.*M), zeros(3, 2)));
assert(isequal(imag(M.*x), zeros(3, 2)));
assert(isequal(imag(x*2), [0; 0; 0]) && isequal(imag(2*x), [0; 0; 0]));
assert(isequal(imag(x./r), [0; 0; 0]));
q = real(r./x);
assert(isnan(q(1)) && q(2) == 3 && q(3) == 0);
assert(isequal(imag(r./x), [0; 0; 0]));
y = complex(NaN, 0);
assert(isreal(y*3) && isnan(y*3) && isreal(3*y) && isreal(y.*r));
assert(isreal(complex(NaN, 0)*complex(2, 0)));
assert(isreal(complex([1; 2], [0; 0]) .* [1 NaN]));

% genuinely complex: imaginary part stays 0 where the operand's is 0
z = complex([1 2], [1 0]);
assert(isequal(imag(z.*[3 NaN]), [3 0]) && isequal(imag([3 NaN].*z), [3 0]));
assert(isequal(imag(z.*[3 Inf]), [3 0]));
assert(real(Inf*complex(0, 1)) == 0 && real(complex(0, 3)/0) == 0);
assert(isnan(imag(z(1)*NaN)) && imag(z(2)*NaN) == 0);
assert(isequal(imag(z./[3 0]), [1/3 0]) && isequal(real(z./[3 0]), [1/3 Inf]));
assert(isequal(imag(z/0), [Inf 0]));
assert(isequal(imag([3; NaN] .\ z.'), [1/3; 0]));
assert(isequal(imag([2 NaN]./z), [-1 0]));
assert(isequal(imag(complex([1 NaN], [2 0])./[0 2]), [Inf 0]));
w = complex([1;2],[1;0]).*[3 NaN];
assert(isequal(imag(w(:, 1)), [3; 0]) && isnan(imag(w(1, 2))) && imag(w(2, 2)) == 0);

% matrix products promote, as in MATLAB
assert(all(isnan(imag(x.'*M))));
assert(isnan(imag(z*[3; NaN])));

disp('SUCCESS')
