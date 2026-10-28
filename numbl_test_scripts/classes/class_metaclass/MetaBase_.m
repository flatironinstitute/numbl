classdef MetaBase_ < handle
    % Fixture for class_metaclass.m.
    properties
        pub = 1
    end
    properties (Access = private)
        priv = 2
    end
    properties (Access = protected)
        prot = 3
    end
    properties (SetAccess = immutable)
        imm = 4
    end
    properties (Hidden)
        hid = 5
    end
    properties (Constant)
        K = 6
    end
    properties (Dependent)
        dep
    end
    properties (Transient)
        tr = 7
    end
    methods
        function v = get.dep(obj)
            v = obj.pub * 10;
        end
        function r = pubMeth(obj, a, b)
            r = a + b;
        end
    end
    methods (Access = private)
        function privMeth(obj)
        end
    end
    methods (Access = protected)
        function protMeth(obj)
        end
    end
    methods (Static)
        function s = statMeth()
            s = 1;
        end
    end
    methods (Hidden)
        function hidMeth(obj)
        end
    end
end
