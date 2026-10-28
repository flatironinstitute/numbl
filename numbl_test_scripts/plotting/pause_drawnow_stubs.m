% pause, drawnow and the other graphics stubs run as statements in a
% JIT-compiled script (they once crashed the JIT's plot bridge).

%!numbl:assert_jit c
x = 1;
pause(0.01);
drawnow;
plot(1:3, [2 4 6]);
xlim([0 4]);
pause(0);
x = x + 1;
assert(x == 2);

disp('SUCCESS');
