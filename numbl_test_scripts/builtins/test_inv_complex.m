% Test complex matrix inversion

% Complex scalar inversion
z = inv(2 + 1i);
expected = (2 - 1i) / 5;
assert(abs(z - expected) < 1e-10, 'complex scalar inv');

% Complex scalar zero should error
try
    inv(0 + 0i);
    error('should have thrown');
catch e
    assert(~isempty(strfind(e.message, 'singular')));
end

% Real scalar inversion
assert(abs(inv(4) - 0.25) < 1e-10, 'real scalar inv');

% Real scalar zero should error
try
    inv(0);
    error('should have thrown');
catch e
    assert(~isempty(strfind(e.message, 'singular')));
end

% Complex 2x2 matrix inversion
A = [1+1i, 2; 3, 4+1i];
B = inv(A);
% A * B should be identity
I_check = A * B;
assert(abs(I_check(1,1) - 1) < 1e-8, 'complex inv identity (1,1)');
assert(abs(I_check(1,2)) < 1e-8, 'complex inv identity (1,2)');
assert(abs(I_check(2,1)) < 1e-8, 'complex inv identity (2,1)');
assert(abs(I_check(2,2) - 1) < 1e-8, 'complex inv identity (2,2)');

% Complex 3x3 matrix inversion
C = [2+1i, 1, 0; 1, 3+2i, 1+1i; 0, 1+1i, 4+1i];
D = inv(C);
I3 = C * D;
for r = 1:3
    for c = 1:3
        if r == c
            assert(abs(I3(r,c) - 1) < 1e-8, sprintf('complex 3x3 inv (%d,%d)', r, c));
        else
            assert(abs(I3(r,c)) < 1e-8, sprintf('complex 3x3 inv (%d,%d)', r, c));
        end
    end
end

% Non-square matrix should error
try
    inv([1 2 3; 4 5 6]);
    error('should have thrown');
catch e
    assert(~isempty(strfind(e.message, 'square')));
end

% Purely imaginary matrix
E = [1i, 2i; 3i, 5i];
F = inv(E);
I_check2 = E * F;
assert(abs(I_check2(1,1) - 1) < 1e-8, 'purely imaginary inv (1,1)');
assert(abs(I_check2(2,2) - 1) < 1e-8, 'purely imaginary inv (2,2)');

disp('SUCCESS')
