% R: an error in a while condition evaluated after the body. The
% interpreter attributes it to the last statement it executed (the body's
% last line), and the JITs record the same per-statement location.
v = [3 2 1];
k = 1;
n = 0;
while v(k) > 0
  n = n + v(k);
  k = k + 1;
end
disp(n)
