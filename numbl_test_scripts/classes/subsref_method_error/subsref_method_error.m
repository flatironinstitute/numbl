% Regression test: when a class overloads subsref, an error thrown from
% inside a REAL method (called as obj.method(...) within another method —
% the dot-in-method bypass) must propagate to the caller. Previously numbl
% caught any error from the method and re-routed the call through subsref,
% masking the real error with a misleading "<name> is not accessible" (the
% bug that hid ultraSEM's initialize error). Verified against MATLAB R2025b.

obj = SR(10);

% invokeCompute(obj) dispatches to the method directly (function-call
% syntax). Inside it, obj.compute() throws SR:internal; that error must
% propagate, not be masked by the subsref overload.
threw = false;
try
    invokeCompute(obj);
catch err
    threw = true;
    assert(~isempty(strfind(err.message, 'compute failed internally')), ...
        ['masked or wrong error: ' err.message]);
end
assert(threw, 'invokeCompute(obj) should have thrown');

% A name handled only by subsref still routes through it.
assert(obj.virtual == 42, 'obj.virtual should route through subsref');

disp('SUCCESS')
