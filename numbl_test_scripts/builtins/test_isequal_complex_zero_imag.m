% isequal compares values only: a complex array whose imaginary parts are
% all zero equals the real array with the same real parts.

a = complex([2; 6], [0; 0]);
assert(isequal(a, [2; 6]));
assert(isequal([2; 6], a));
assert(isequal(a, a, [2; 6]));
assert(isequal(complex(2, 0), 2));
assert(~isequal(complex([2; 6], [0; 1]), [2; 6]));
assert(~isequal(complex([2; 6], [NaN; 0]), [2; 6]));
assert(~isequal(complex([2; 6], [0; 0]), [2; 7]));
assert(isequal(complex(5, 0), 5));

% inside compiled functions: real vs complex tensors and scalars
assert(eqt([1 2], [1 2] + 0i));
assert(~eqt([1 2], [1 2] + 1i));
assert(eqt([1 2] + 2i, [1 2] + 2i));
assert(eqs(3, 3 + 0i));
assert(eqs(3 + 0i, 3));
assert(~eqs(3, 3 + 1i));
assert(eqst(3 + 1i, 3 + 1i));
assert(~eqst(3, [3 3] + 0i));

disp('SUCCESS')

function r = eqt(x, y)
  r = isequal(x, y);
end

function r = eqs(x, y)
  t = [x x];
  r = isequal(t(1), y);
end

function r = eqst(s, t)
  r = isequal(s, t) && isequal(t, s);
end
