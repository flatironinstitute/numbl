% fplot3 example - 3D parametric curve
fplot3(@(t) sin(t), @(t) cos(t), @(t) t/5, [0 6*pi]);
title('3D helix');
