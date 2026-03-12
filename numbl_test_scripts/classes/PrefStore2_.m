classdef PrefStore2_
% Test class mirroring chebfunpref pattern more closely:
% Constructor assigns to prefList directly, and subsasgn does NOT
% have a special case for 'prefList' — it falls through to an else
% branch that accesses obj.prefList.techPrefs.

    properties
        prefList
    end

    methods
        function obj = PrefStore2_()
            % Direct assignment to prefList (like chebfunpref constructor)
            obj.prefList = struct();
            obj.prefList.alpha = 10;
            obj.prefList.techPrefs = struct();
            obj.prefList.techPrefs.x = 1;
            obj.prefList.techPrefs.y = 2;
        end

        function out = subsref(obj, S)
            switch S(1).type
                case '.'
                    if strcmp(S(1).subs, 'prefList')
                        out = obj.prefList;
                        if numel(S) > 1
                            out = builtin('subsref', out, S(2:end));
                        end
                    elseif isfield(obj.prefList, S(1).subs)
                        out = obj.prefList.(S(1).subs);
                        if numel(S) > 1
                            out = builtin('subsref', out, S(2:end));
                        end
                    else
                        error('No field %s', S(1).subs);
                    end
                otherwise
                    error('Unsupported subscript type');
            end
        end

        function obj = subsasgn(obj, S, val)
            % Like chebfunpref: no special case for 'prefList' property.
            % Known fields go to prefList, unknown go to prefList.techPrefs.
            switch S(1).type
                case '.'
                    if isfield(obj.prefList, S(1).subs)
                        obj.prefList = builtin('subsasgn', obj.prefList, S, val);
                    else
                        obj.prefList.techPrefs = builtin('subsasgn', ...
                            obj.prefList.techPrefs, S, val);
                    end
                otherwise
                    error('Unsupported subscript type');
            end
        end
    end
end
