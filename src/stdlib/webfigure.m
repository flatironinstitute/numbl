function webfigure(d)
%WEBFIGURE Display a self-contained "directory figure" in the figure pane.
%   WEBFIGURE(D) renders a directory of static files as an interactive view,
%   bypassing the standard plotting (axes/traces) entirely. D is a
%   containers.Map from relative file path (char) to file content: char for
%   text (HTML/JS/CSS/JSON, ...) or uint8 for binary. The map must include an
%   'index.html' entry, which is loaded in an iframe.
%
%   The files are served locally to the figure (no disk, no upload), so any
%   self-contained static site works unchanged.
%
%   Example:
%       d = containers.Map();
%       d('index.html') = '<h1>Hello from a numbl directory figure</h1>';
%       webfigure(d);
    ks = keys(d);
    contents = cell(1, numel(ks));
    for i = 1:numel(ks)
        contents{i} = d(ks{i});
    end
    drawwebfigure(ks, contents);
end
