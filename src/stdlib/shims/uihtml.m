classdef uihtml < handle
    %UIHTML Create an HTML UI component (numbl subset).
    %   H = UIHTML('HTMLSource', HTML) renders the self-contained HTML string
    %   HTML in the figure pane. An optional leading parent argument (e.g. a
    %   figure) is accepted and ignored. Supported name-value options:
    %   HTMLSource, Data, Position.
    %
    %   H = UIHTML('HTMLSource', HTML, 'Data', X) also sends X into the page.
    %   Mirroring MATLAB, X is encoded with jsonencode, parsed in the page with
    %   JSON.parse, and set on the JavaScript `htmlComponent.Data` object,
    %   firing any "DataChanged" listener registered in the page's
    %   `function setup(htmlComponent)`. To update the data after construction,
    %   set H.Data and call show(H) (numbl re-renders the component):
    %
    %       h = uihtml('HTMLSource', html, 'Data', struct('n', 1));
    %       h.Data = struct('n', 2);
    %       show(h);
    %
    %   numbl currently supports HTMLSource given as an HTML markup string (a
    %   single self-contained document). HTML file paths, supporting files, and
    %   the JavaScript-to-MATLAB reverse channel (DataChangedFcn,
    %   HTMLEventReceivedFcn, sendEventToMATLAB) are not yet supported.
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
                if isempty(obj.Data)
                    drawuihtml(obj.HTMLSource);
                else
                    drawuihtml(obj.HTMLSource, jsonencode(obj.Data));
                end
            end
        end

        function show(obj)
            if isempty(obj.Data)
                drawuihtml(obj.HTMLSource);
            else
                drawuihtml(obj.HTMLSource, jsonencode(obj.Data));
            end
        end
    end
end
