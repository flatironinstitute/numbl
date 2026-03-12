classdef PrefStore_
% Test class with overloaded subsref/subsasgn.
% Mirrors the pattern used by chebfunpref: public dot-access on the
% class routes through an internal prefList struct.
    properties
        prefList
    end

    methods
        function obj = PrefStore_()
            obj.prefList.alpha = 10;
            obj.prefList.opts = struct();
            obj.prefList.opts.x = 1;
            obj.prefList.opts.y = 2;
            obj.prefList.opts.z = 3;
        end

        function out = subsref(obj, S)
            % Route dot-access through prefList
            switch S(1).type
                case '.'
                    if strcmp(S(1).subs, 'prefList')
                        % Direct access to prefList (used internally)
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
            % Route dot-assignment through prefList
            switch S(1).type
                case '.'
                    if strcmp(S(1).subs, 'prefList')
                        % Direct assignment to prefList
                        if numel(S) > 1
                            obj.prefList = builtin('subsasgn', obj.prefList, S(2:end), val);
                        else
                            obj.prefList = val;
                        end
                    elseif isfield(obj.prefList, S(1).subs)
                        % Redirect: obj.X.Y... = val becomes
                        % obj.prefList = builtin('subsasgn', obj.prefList, S, val)
                        obj.prefList = builtin('subsasgn', obj.prefList, S, val);
                    else
                        error('No field %s', S(1).subs);
                    end
                otherwise
                    error('Unsupported subscript type');
            end
        end
    end
end
