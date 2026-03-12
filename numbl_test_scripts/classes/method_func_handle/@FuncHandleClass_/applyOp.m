function result = applyOp(obj, op)
% Apply operation op to obj.value, wrapping the result in a new instance.
% The op function handle may reference a local function from the caller's
% file — that local function must still be resolvable via dispatch even
% though this method is in a different file.
    result = FuncHandleClass_(op(obj.value));
end
