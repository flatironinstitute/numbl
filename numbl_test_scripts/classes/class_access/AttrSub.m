classdef AttrSub < AttrBase
    % Fixture for class_access.m: subclass access to protected and private
    % members of AttrBase.
    methods
        function obj = AttrSub(setImm)
            obj@AttrBase(1);
            if nargin > 0 && setImm
                obj.imm = 9;
            end
        end
        function v = readProt(obj)
            v = obj.prot;
        end
        function v = readProtOf(obj, other)
            v = other.prot;
        end
        function v = readPriv2(obj)
            v = obj.priv;
        end
        function v = callProt(obj)
            v = obj.protMeth() + protMeth(obj);
        end
        function v = callPriv2(obj)
            v = obj.privMeth();
        end
        function setSprot(obj, v)
            obj.sprot = v;
        end
        function setRo2(obj, v)
            obj.ro = v;
        end
    end
end
