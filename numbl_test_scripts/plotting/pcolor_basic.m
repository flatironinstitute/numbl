% Basic pcolor smoke test: rectangular pseudocolor plot from a single matrix.

C = magic(5);
pcolor(C);
colorbar;

disp('SUCCESS');
