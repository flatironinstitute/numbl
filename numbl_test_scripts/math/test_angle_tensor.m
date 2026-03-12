% Test angle() on tensors (real and complex)

% angle on real tensor
ang_real = angle([1, -1, 0]);
assert(ang_real(1) == 0, 'angle positive');
assert(abs(ang_real(2) - pi) < 1e-10, 'angle negative');
assert(ang_real(3) == 0, 'angle zero');

% angle on complex tensor
z = [1+1i, 3+4i, -1+0i];
ang = angle(z);
assert(abs(ang(1) - pi/4) < 1e-10, 'angle 1+1i');
assert(abs(ang(2) - atan2(4, 3)) < 1e-10, 'angle 3+4i');
assert(abs(ang(3) - pi) < 1e-10, 'angle -1');

% angle on real matrix
M = [1 -1; 0 2];
ang_m = angle(M);
assert(ang_m(1,1) == 0, 'angle matrix (1,1)');
assert(abs(ang_m(1,2) - pi) < 1e-10, 'angle matrix (1,2)');

disp('SUCCESS');
