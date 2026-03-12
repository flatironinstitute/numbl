classdef (InferiorClasses = {?BasicObj_}) SuperiorObj_
    properties
        val
    end
    methods
        function obj = SuperiorObj_(v)
            obj.val = v;
        end
        function r = plus(a, b)
            % Returns 'superior' to identify which plus was called
            r = 'superior';
        end
        function r = minus(a, b)
            % Test that minus also dispatches correctly
            % Internally calls plus, mimicking adchebfun pattern
            r = plus(a, b);
        end
        function r = mtimes(a, b)
            r = 'superior';
        end
    end
end
