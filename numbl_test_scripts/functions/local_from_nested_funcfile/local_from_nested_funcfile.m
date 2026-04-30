% Reproduces the chunkerfit pattern: a function file (do_fit.m) defines a main
% function with a NESTED function, plus sibling LOCAL helper functions. The
% nested function is returned as a handle (or passed through an indirection
% function that calls it back), and from inside the nested function we call
% one of the sibling local helpers (myppval-style).

% Test 1: Direct handle to the nested function, called from this script.
h = do_fit(2, 3);
out = h(4);
% inner returns combine_local(a,b,z) + add_local(a,b)
%   = (a*z + b) + (a + b) = (2*4 + 3) + (2 + 3) = 11 + 5 = 16
assert(out == 16)

% Test 2: Pass the nested-function handle through an indirection that calls it
% (mimicking chunkerfit handing @splinefunc to chunkerfunc, which then invokes
% it from a totally different scope).
out2 = apply_handle(h, 5);
%   = (2*5 + 3) + (2 + 3) = 13 + 5 = 18
assert(out2 == 18)

% Test 3: Nested function with multiple outputs, also calling a local helper.
[r1, r2] = do_fit_multi(10);
% inner returns [scale_local(x, 2), scale_local(x, 3)]
%   = [10*2, 10*3] = [20, 30]
assert(r1 == 20)
assert(r2 == 30)

disp('SUCCESS')
