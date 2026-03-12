function result = compute(obj, op, data)
% External method with local helper functions.
% The call to outerHelper with a function handle has arg types that
% can't be fully resolved at compile time, forcing JIT.
% Inside outerHelper, the call to innerHelper also has unknown types,
% so it must also JIT-compile and find the local function via methodScope.
    result = outerHelper(op, data);
end

function result = outerHelper(op, data)
% When this function is JIT-compiled (via dispatchUnknown from compute),
% calls to innerHelper must propagate the methodName in the scope so
% the JIT callback can find innerHelper via withMethodScope.
    if iscell(data)
        val = data{1};
    else
        val = data;
    end
    result = innerHelper(op, val);
end

function result = innerHelper(op, data)
    result = op(data.x) + data.y;
end
