function result = processVals(obj)
% Creates an anonymous function that captures a local helper, then passes
% it to applyOp. When applyOp calls the anonymous function from a
% different file context, the local helper must still be resolvable.
    op = @(x) doubleIt(x);
    result = applyOp(obj, op);
end

function y = doubleIt(x)
    y = x * 2;
end
