classdef Vector2D_
    properties
        X
        Y
    end
    methods
        function obj = Vector2D_(x, y)
            obj.X = x;
            obj.Y = y;
        end
        function r = magnitude(obj)
            r = sqrt(obj.X^2 + obj.Y^2);
        end
        function r = add(obj, other)
            r = Vector2D_(obj.X + other.X, obj.Y + other.Y);
        end
        function r = scale(obj, factor)
            r = Vector2D_(obj.X * factor, obj.Y * factor);
        end
        function r = normalized(obj)
            m = obj.magnitude();
            if m == 0
                r = Vector2D_(0, 0);
            else
                r = obj.scale(1 / m);
            end
        end
        function r = dot_prod(obj, other)
            r = obj.X * other.X + obj.Y * other.Y;
        end
        function r = project_onto(obj, other)
            % Method calling multiple other methods + constructor
            d = obj.dot_prod(other);
            m2 = other.dot_prod(other);
            factor = d / m2;
            r = other.scale(factor);
        end
        function [m, a] = polar(obj)
            % Multiple return values from a method
            m = obj.magnitude();
            a = atan2(obj.Y, obj.X);
        end
    end
end
