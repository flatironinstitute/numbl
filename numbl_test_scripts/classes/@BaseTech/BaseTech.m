classdef BaseTech
    properties
        coeffs
    end
    methods
        function obj = BaseTech(c)
            if nargin > 0
                obj.coeffs = c;
            else
                obj.coeffs = 0;
            end
        end
    end
    methods (Static = true)
        % Static method: evaluates coefficients at points using simple dot product
        out = evaluate(x, coeffs)
    end
    methods
        function h = getTechHandle(~)
            % Returns @BaseTech from inside a BaseTech method.
            % This tests that @ClassName inside ClassName's own method
            % produces a valid function handle with the class name preserved.
            h = @BaseTech;
        end
    end
end
