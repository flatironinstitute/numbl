classdef SepVec
    properties
        X
        Y
        Z
    end
    methods
        function obj = SepVec(x, y, z)
            obj.X = x;
            obj.Y = y;
            obj.Z = z;
        end
        function r = magnitude(obj)
            r = sqrt(obj.X^2 + obj.Y^2 + obj.Z^2);
        end
        function r = scale(obj, f)
            r = SepVec(obj.X * f, obj.Y * f, obj.Z * f);
        end
    end
end
