classdef InnerObj
    properties
        coeffs
    end
    methods
        function obj = InnerObj(c)
            if nargin > 0
                obj.coeffs = c;
            end
        end
        function obj = uminus(obj)
            obj.coeffs = -obj.coeffs;
        end
    end
end
