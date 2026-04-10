% Smoke-test colorbar at various locations. Just verifies no errors are
% thrown — visual verification is manual.

C = peaks(20);

figure(1); clf
pcolor(C); colorbar('eastoutside');

figure(2); clf
pcolor(C); colorbar('westoutside');

figure(3); clf
pcolor(C); colorbar('northoutside');

figure(4); clf
pcolor(C); colorbar('southoutside');

% colorbar('off') should be a no-op (no error)
figure(5); clf
pcolor(C); colorbar; colorbar('off');

disp('SUCCESS');
