% R: an out-of-bounds read in a JIT'd loop, uncaught. The JITs raise the
% interpreter's error ("Index exceeds array bounds") attributed to the
% statement's line, and keep the output printed before the error.
v = [1 2 3];
s = 0;
for k = 1:5
  fprintf('k=%d\n', k);
  s = s + v(k);
end
disp(s)
