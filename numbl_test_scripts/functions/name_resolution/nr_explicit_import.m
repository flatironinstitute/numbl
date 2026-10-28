function r = nr_explicit_import()
    % Dotted names through `import nrpkg.NrPkgClass`.
    import nrpkg.NrPkgClass
    r.static = NrPkgClass.thrice(2);
    h = @NrPkgClass.thrice;
    r.staticHandle = h(5);
    r.className = class(NrPkgClass());
    r.existClass = exist('NrPkgClass');
end
