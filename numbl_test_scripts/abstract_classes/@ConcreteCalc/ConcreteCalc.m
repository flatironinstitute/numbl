classdef ConcreteCalc < BaseCalc
    methods
        function obj = ConcreteCalc(v)
            obj = obj@BaseCalc(v);
            % Call parent method which internally calls obj.compute() (abstract static)
            [obj, r] = populate(obj, 3, 4);
            obj.value = r;
        end
    end
    methods (Static)
        function result = compute(x, y)
            result = x + y;
        end
    end
end
