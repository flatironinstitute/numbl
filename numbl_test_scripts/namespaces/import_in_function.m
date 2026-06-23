% A wildcard import inside a function body must let a bare call resolve to
% the package function (imports are function-scoped in MATLAB).

assert(impwild.branch(5) == 11, 'function-scoped wildcard import');

disp('SUCCESS')
