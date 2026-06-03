classdef uihtml < handle
    %UIHTML Create an HTML UI component (numbl subset).
    %   H = UIHTML('HTMLSource', HTML) renders the self-contained HTML string
    %   HTML in the figure pane. An optional leading parent argument (e.g. a
    %   figure) is accepted and ignored. Supported name-value options:
    %   HTMLSource, Data, Position.
    %
    %   numbl currently supports HTMLSource given as an HTML markup string (a
    %   single self-contained document). The component renders when HTMLSource
    %   is set at construction; call show(h) to re-render after changing it.
    %   (HTML file paths, supporting files, and the Data/event bridge are not
    %   yet supported.)
    properties
        HTMLSource = ''
        Data = []
        Position = [100 100 100 100]
    end
    methods
        function obj = uihtml(varargin)
            args = varargin;
            % Ignore an optional leading parent argument (anything that is not
            % a name-value name string).
            if numel(args) >= 1 && ~(ischar(args{1}) || isstring(args{1}))
                args = args(2:end);
            end
            for i = 1:2:numel(args) - 1
                name = args{i};
                val = args{i + 1};
                if strcmpi(name, 'HTMLSource')
                    obj.HTMLSource = val;
                elseif strcmpi(name, 'Data')
                    obj.Data = val;
                elseif strcmpi(name, 'Position')
                    obj.Position = val;
                end
            end
            if ~isempty(obj.HTMLSource)
                drawuihtml(obj.HTMLSource);
            end
        end

        function show(obj)
            drawuihtml(obj.HTMLSource);
        end
    end
end
