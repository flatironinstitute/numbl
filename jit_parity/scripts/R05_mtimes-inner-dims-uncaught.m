% R: a matrix product whose inner dimensions only disagree at run time,
% uncaught: "Inner matrix dimensions must agree: 3 vs 2".
B = ones(3, 2);
t = 0;
for k = 3:-1:2
  A = ones(2, k);
  C = A * B;
  t = t + sum(C(:));
  fprintf('t=%g\n', t);
end
