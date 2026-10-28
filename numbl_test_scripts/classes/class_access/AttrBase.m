classdef AttrBase < handle
    % Fixture for class_access.m: properties and methods with every access
    % attribute.
    properties
        pub = 1
    end
    properties (Access = private)
        priv = 2
    end
    properties (Access = protected)
        prot = 3
    end
    properties (SetAccess = private)
        ro = 4
    end
    properties (SetAccess = immutable)
        imm = 5
    end
    properties (GetAccess = private, SetAccess = public)
        wo = 6
    end
    properties (SetAccess = protected)
        sprot = 7
    end
    properties (Dependent)
        dep
    end
    properties (Access = {?AttrFriend})
        friendly = 8
    end
    methods
        function obj = AttrBase(x)
            if nargin > 0
                obj.imm = x;
                obj.imm = x + 1;
            end
        end
        function v = get.dep(obj)
            v = obj.pub * 10;
        end
        function v = readPriv(obj)
            v = obj.priv;
        end
        function v = readWo(obj)
            v = obj.wo;
        end
        function setPriv(obj, v)
            obj.priv = v;
        end
        function setRo(obj, v)
            obj.ro = v;
        end
        function setImm(obj, v)
            obj.imm = v;
        end
        function v = callPriv(obj)
            v = obj.privMeth() + privMeth(obj);
        end
        function f = privGetter(obj)
            f = @() obj.priv;
        end
        function v = viaLocal(obj)
            v = localReadPriv(obj);
        end
        function v = readFriendly(obj)
            v = obj.friendly;
        end
    end
    methods (Access = private)
        function v = privMeth(obj)
            v = 42;
        end
    end
    methods (Access = protected)
        function v = protMeth(obj)
            v = 43;
        end
    end
    methods (Static, Access = private)
        function v = privStatic()
            v = 44;
        end
    end
    methods (Static)
        function v = callPrivStatic()
            v = AttrBase.privStatic();
        end
    end
end

function v = localReadPriv(obj)
% Local functions of a classdef file have the class's access.
v = obj.priv;
end
