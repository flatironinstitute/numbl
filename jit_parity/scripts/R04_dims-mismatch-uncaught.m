% R: an element-wise operation on tensors whose shapes only disagree at
% run time, uncaught: "Matrix dimensions must agree: [1,2] vs [1,3]".
b = ones(1, 3);
acc = 0;
for k = 3:-1:1
  a = ones(1, k);
  c = a + b;
  acc = acc + sum(c);
  fprintf('acc=%g\n', acc);
end
