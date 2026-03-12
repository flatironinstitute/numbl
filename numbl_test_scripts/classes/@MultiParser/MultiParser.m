classdef MultiParser
    properties
        val1
        val2
        val3
        val4
        val5
    end
    methods
        function obj = MultiParser(varargin)
            if nargin == 0
                obj.val1 = 0;
                obj.val2 = 0;
                obj.val3 = 0;
                obj.val4 = 0;
                obj.val5 = 0;
                return
            end
            % Spread varargin into local parseInputs (5 outputs)
            [a, b, c, d, e] = parseInputs(varargin{:});
            obj.val1 = a;
            obj.val2 = b;
            obj.val3 = c;
            obj.val4 = d;
            obj.val5 = e;
        end

        function r = compute(obj, varargin)
            % This method also has a local parseInputs but with 3 outputs
            [x, y, z] = parseComputeInputs(varargin{:});
            r = obj.val1 + x + y + z;
        end
    end
end

% Local parseInputs for the constructor - 5 outputs
function [a, b, c, d, e] = parseInputs(op, extra)
    if nargin < 2
        extra = 100;
    end
    a = op;
    b = extra;
    c = op + extra;
    d = op * 2;
    e = extra * 2;
end

% Local parseComputeInputs for compute method - 3 outputs
function [x, y, z] = parseComputeInputs(a, b)
    if nargin < 2
        b = 1;
    end
    x = a;
    y = b;
    z = a + b;
end
