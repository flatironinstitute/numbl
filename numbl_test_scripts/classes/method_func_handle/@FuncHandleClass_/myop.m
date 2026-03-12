function result = myop(obj)
% Class method: wraps the result of doWork in a new instance.
% Passes @myop as a function handle. When doWork calls @myop with a
% plain number, it should dispatch to the workspace myop.m (not this
% class method), because @myop should use runtime dispatch.
    result = FuncHandleClass_(doWork(obj.value, @myop));
end
