classdef SimpleCounter_
    properties
        Value = 0
        Step = 1
    end
    methods
        function obj = SimpleCounter_(val, step)
            if nargin >= 1
                obj.Value = val;
            end
            if nargin >= 2
                obj.Step = step;
            end
        end
        function obj = increment(obj)
            obj.Value = obj.Value + obj.Step;
        end
        function obj = increment_n(obj, n)
            for i = 1:n
                obj = obj.increment();
            end
        end
        function r = to_configured(obj)
            % Constructor calls: return a different class instance
            if obj.Value > 50
                r = ConfiguredObj_('fast');
            else
                r = ConfiguredObj_('quality');
            end
        end
        function r = equals(obj, other)
            r = obj.Value == other.Value && obj.Step == other.Step;
        end
    end
end
