function r = nr_wildcard_import()
    % Dotted and bare names under `import nrpkg.*`.
    import nrpkg.*
    r.static = NrPkgClass.thrice(2);
    h = @NrPkgClass.thrice;
    r.staticHandle = h(5);
    r.fn = nr_pkgfn(1);
    g = @nr_pkgfn;
    r.fnHandle = g(2);
    r.subpackage = inner.nr_innerfn(1);
    r.feval = feval('nr_pkgfn', 4);
    r.emptySize = size(NrPkgClass.empty(0, 1));
    % exist does not report package functions.
    r.existFn = exist('nr_pkgfn');
    r.whichFnFound = ~isempty(which('nr_pkgfn'));
end
