classdef SR
    properties
        data = 0
    end
    methods
        function obj = SR(x)
            if nargin > 0
                obj.data = x;
            end
        end
        function varargout = subsref(obj, s)
            % Custom subsref: synthesize a 'virtual' field, otherwise error.
            if strcmp(s(1).type, '.')
                name = s(1).subs;
                if strcmp(name, 'virtual')
                    varargout = {42};
                    return
                end
                error('SR:badName', '%s is not accessible via subsref', name);
            end
            error('SR:badType', 'unsupported index type');
        end
        function y = compute(obj) %#ok<STOUT>
            % A real method that throws its own internal error.
            error('SR:internal', 'compute failed internally');
        end
        function y = invokeCompute(obj)
            % Calls a sibling method via dot from inside a method body (the
            % dot-in-method bypass). The real error from compute must surface
            % here, not be masked by the subsref overload.
            y = obj.compute();
        end
    end
end
