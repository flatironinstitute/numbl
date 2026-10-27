% TEST: a plot call in a compiled script draws instead of printing.
%
% opt2 (C-JIT) wrote each plot call as a raw "\x1emtoc2:plot\t{...}" line
% straight to stdout, bypassing the host output hook, so the record was
% printed (and no plot instruction recorded) while opt0/opt1 printed only
% the fprintf output. Char arguments were also encoded as null.
x = 1:3;
plot(x, x.^2, 'r-');
title('squares');
fprintf('done %d\n', numel(x));
