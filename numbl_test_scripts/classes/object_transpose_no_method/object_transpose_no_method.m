% Test: transpose (.') and ctranspose (') on a class instance that defines
% neither method. MATLAB's built-in object-array transpose rearranges the
% array elements; for a scalar object it is an identity. numbl must not try
% to transpose the object as if it were a numeric value.

a = TBox(7);

% Non-conjugate transpose
b = a.';
assert(b.getv() == 7, 'Expected 7 from a.''');

% Conjugate transpose
c = a';
assert(c.getv() == 7, 'Expected 7 from a''');

% Inside an expression (mimics chebfun/surfacefun mtimes using x.')
d = a.'.';
assert(d.getv() == 7, 'Expected 7 from a.''.''');

disp('SUCCESS')
