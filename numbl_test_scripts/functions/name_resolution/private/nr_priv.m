function r = nr_priv(x)
    % Calls its own local function and a sibling private function by name,
    % by handle and through arrayfun; exist/which/nargin see both.
    r.fevalLocal = feval('nr_priv_loc', x);
    r.arrayfunLocal = arrayfun(@nr_priv_loc, [x x]);
    r.fevalSibling = feval('nr_priv2', x);
    r.cellfunSibling = cellfun(@nr_priv2, {x});
    r.existLocal = exist('nr_priv_loc');
    r.existSibling = exist('nr_priv2');
    r.whichFound = ~isempty(which('nr_priv_loc')) && ~isempty(which('nr_priv2'));
    r.narginLocal = nargin('nr_priv_loc');
end

function y = nr_priv_loc(x, unused)
    y = x + 100;
end
