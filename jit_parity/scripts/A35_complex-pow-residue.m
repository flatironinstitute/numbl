% TEST: complex power a^b with NONZERO base (distinct from the known/"done"
% cpow zero-base case). C-JIT (opt2) gives a cleaner/exact result; interpreter
% and JS-JIT carry a floating-point residue from exp(b*log(a)).
%
% (-1 - 1i)^2:       opt0 -3.6739e-16 + 2i      | opt1 same | opt2 2i          <-- DIVERGES
% (-1)^(0.5 + 1i):   opt0 2.6461e-18 + 0.043214i| opt1 same | opt2 0.043214i   <-- DIVERGES
%
% DIVERGING MODE: opt2 vs opt0/opt1.
% CAUSE: interpreter & JS mtoc2_cpow compute a^b = exp(b*log(a)), which leaves
%   a tiny nonzero residue in the lane that should be exactly 0. C99 cpow
%   returns the cleaner value (exact 2i / pure-imaginary).
% JIT ENGAGEMENT: confirmed (cpow in --dump-c).
disp((-1 - 1i)^2);
disp((-1) ^ (0.5 + 1i));
