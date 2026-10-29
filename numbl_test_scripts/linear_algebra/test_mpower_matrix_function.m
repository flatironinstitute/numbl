% mpower beyond integer powers: errors for non-square operands, and
% scalar^matrix / matrix^non-integer computed as matrix functions.

msg = ['Incorrect dimensions for raising a matrix to a power. Check that ' ...
       'the matrix is square and the power is a scalar. To operate on each ' ...
       'element of the matrix individually, use POWER (.^) for elementwise power.'];
bad = {'[1 2 3; 4 5 6]^2', '[1 2 3; 4 5 6]^0.5', '[1 2; 3 4]^[1 2; 3 4]', ...
       '[1 2; 3 4]^[1 2]', '2^[1 2 3; 4 5 6]', '[1 2]^[1 2]', '[1 2 3]^2', ...
       'zeros(0, 3)^2', 'mpower([1 2; 3 4], [1; 2])', '''ab''^2'};
for k = 1:numel(bad)
    try
        eval([bad{k} ';']);
        error('no error for %s', bad{k});
    catch e
        assert(strcmp(e.identifier, 'MATLAB:mpower:notScalarAndSquareMatrix'), bad{k});
        assert(strcmp(e.message, msg));
    end
end
try
    ones(2, 2, 2)^2;
    error('no error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:mpower:inputsMustBe2D'));
end

close = @(X, Y) isequal(size(X), size(Y)) && max(abs(X(:) - Y(:))) < 1e-10 * max(1, max(abs(Y(:))));

% scalar ^ matrix
B = 2^[1 2; 3 4];
assert(isreal(B) && close(B, [10.4827393896286 14.1518788283206;21.227818242481 31.7105576321096]));
B = 2^[0 1; -1 0];
assert(isreal(B) && close(B, [0.769238901363972 0.638961276313635;-0.638961276313635 0.769238901363972]));
B = 2^[2 1; 1 2];
assert(close(B, [5 3; 3 5]));
B = 1.5^[1 2; 3 4];
assert(close(B, [2.7640684339002 2.77519660818208;4.16279491227312 6.92686334617332]));
B = (-2)^[1 2; 3 4];
assert(~isreal(B) && close(B, [-3.6348357937872-9.65017281689718i -5.73725937821016-13.0278996072534i;-8.60588906731524-19.54184941088i -12.2407248611024-29.1920222277772i]));
B = complex(2, 1)^[1 2; 3 4];
assert(close(B, [-13.7803575357114+10.8183670686095i -21.1479562052496+15.952497149505i;-31.7219343078745+23.9287457242575i -45.5022918435859+34.747112792867i]));

% matrix ^ non-integer or complex scalar
B = [2 1; 1 2]^0.5;
assert(isreal(B) && close(B, [1.36602540378444 0.366025403784439;0.366025403784439 1.36602540378444]));
assert(close(B * B, [2 1; 1 2]));
B = [4 0; 0 9]^0.5;
assert(isreal(B) && close(B, [2 0; 0 3]));
B = [0 1; -1 0]^0.5;
assert(isreal(B) && close(B, [0.707106781186547 0.707106781186547;-0.707106781186547 0.707106781186547]));
B = [2 1; 1 2]^-0.5;
assert(close(B, [0.788675134594813 -0.211324865405187;-0.211324865405187 0.788675134594813]));
B = [1 2; 3 4]^0.5;
assert(~isreal(B) && close(B, [0.553688567145911+0.464394162839071i 0.806960727013216-0.212426478766402i;1.21044109051982-0.318639718149603i 1.76412965766574+0.145754444689468i]));
B = [1 2; 3 4]^2.5;
assert(close(B, [15.9802308752196+0.0643619583774646i 23.2900216657499-0.0294409044701374i;34.9350324986248-0.0441613567052062i 50.9152633738444+0.0202006016722584i]));
B = [1 2; 3 4]^(1+1i);
assert(close(B, [-0.148203876576107+1.28575268298682i -0.203094266645724+1.85431317151839i;-0.304641399968586+2.78146975727759i -0.452845276544693+4.06722244026441i]));
B = [1+1i 2; 3 4]^0.5;
assert(close(B, [0.960358756797274+0.716531091101355i 0.656211635264416-0.191124222478304i;0.984317452896623-0.286686333717455i 1.84911409845475+0.101738939751692i]));
B = [1 2; 3 4]^NaN;
assert(all(isnan(B(:))) && isequal(size(B), [2 2]));

% integer powers
assert(close([1 2; 3 4]^-2, [5.5 -2.5; -3.75 1.75]));
assert(isequal([1+1i 2; 3 4]^3, [34+14i 52+12i; 78+18i 118+6i]));
assert(isequal(true(2)^2, [2 2; 2 2]));
assert(isequal([1 2; 3 4]^true, [1 2; 3 4]));
assert(isequal(size(zeros(0, 0)^0.5), [0 0]));
assert(isequal(size(2^zeros(0, 0)), [0 0]));
A = [1 1; 0 1];
assert(isequal(A^10, [1 10; 0 1]));

disp('SUCCESS');
