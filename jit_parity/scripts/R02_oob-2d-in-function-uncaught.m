% R: a per-axis out-of-bounds read inside a JIT'd function (call shape),
% uncaught. The error is reported with the function's call frame
% ("Error using f (file:line)" ... "Error in file:line").
M = reshape(1:6, 2, 3);
disp(colsum(M, 3));
disp(colsum(M, 4));

function s = colsum(M, n)
  s = 0;
  for j = 1:n
    for i = 1:2
      s = s + M(i, j);
    end
  end
end
