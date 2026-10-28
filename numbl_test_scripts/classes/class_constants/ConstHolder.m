classdef ConstHolder
    % Fixture for class_constants.m: Constant properties whose defaults call
    % static methods (including a private one) and refer to each other.
    properties (Constant)
        K = ConstHolder.makeK()
        K2 = ConstHolder.K * 2
        V = [10 20 30]
        C = {1, 'two'}
        S = struct('a', 4)
        H = ConstHandle_()
    end
    properties (Constant, Access = private)
        PK = ConstHolder.helper(3)
    end
    properties
        x
    end
    methods
        function obj = ConstHolder(x)
            % The constructor needs an argument: ConstHolder.K must not
            % call it.
            obj.x = x;
        end
        function v = sumK(obj)
            v = obj.K + ConstHolder.K2 + obj.PK;
        end
    end
    methods (Static)
        function k = makeK()
            ConstHolder.counter(1);
            k = 7;
        end
        function n = counter(inc)
            persistent count
            if isempty(count)
                count = 0;
            end
            count = count + inc;
            n = count;
        end
        function v = getPK()
            v = ConstHolder.PK;
        end
    end
    methods (Static, Access = private)
        function v = helper(x)
            v = x * 100;
        end
    end
end
