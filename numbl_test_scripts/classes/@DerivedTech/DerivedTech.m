classdef DerivedTech < BaseTech
    methods
        function obj = DerivedTech(c)
            obj = obj@BaseTech(c);
        end
        function y = callEval(obj, x)
            % Call inherited static method via instance
            y = obj.evaluate(x, obj.coeffs);
        end
    end
end
