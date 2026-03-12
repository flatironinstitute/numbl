% Test airy with complex arguments

%% Scalar complex: Ai(z)
z = exp(2*pi*1i/3);
val = airy(0, z);
assert(abs(val - (0.556652857257180 - 0.243272564005050i)) < 1e-10, 'airy(0,z)');

%% Scalar complex: Ai'(z)
val = airy(1, z);
assert(abs(val - (-0.443543463183998 - 0.164196119793036i)) < 1e-10, 'airy(1,z)');

%% Scalar complex: Bi(z)
val = airy(2, z);
assert(abs(val - (0.477605902937721 + 0.421360440944298i)) < 1e-10, 'airy(2,z)');

%% Scalar complex: Bi'(z)
val = airy(3, z);
assert(abs(val - (0.439847574013667 - 0.284396021887204i)) < 1e-10, 'airy(3,z)');

%% Default n=0
val = airy(z);
assert(abs(val - (0.556652857257180 - 0.243272564005050i)) < 1e-10, 'airy(z) default');

%% Pure imaginary
val = airy(1i);
assert(abs(val - (0.331493305432141 - 0.317449858968444i)) < 1e-10, 'airy(1i)');

%% Complex tensor
zz = [1+1i, 2-1i; -1+2i, 0.5i];
vals = airy(zz);
expected = [0.060458308371838 - 0.151889565877181i, ...
            0.001697766857265 + 0.040718017053224i; ...
            1.695064089797038 - 1.424184559346541i, ...
            0.353649223375102 - 0.136802054228524i];
assert(max(abs(vals(:) - expected(:))) < 1e-10, 'airy complex tensor');

%% airy(1, complex_tensor)
vals1 = airy(1, zz);
expected1 = [-0.130627953499647 + 0.163067596449324i, ...
             -0.015110279283227 - 0.062458954713600i; ...
             -2.867653785707360 - 0.872491772861567i, ...
             -0.303140780165206 + 0.011153850054973i];
assert(max(abs(vals1(:) - expected1(:))) < 1e-10, 'airy(1, complex tensor)');

disp('SUCCESS');
