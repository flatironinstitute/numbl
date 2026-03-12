function result = doWork(val, op)
% Calls the op function handle with a plain number.
% op should be @myop, which at runtime should dispatch to the workspace
% myop.m (for numbers), not the class method.
    result = op(val);
end
