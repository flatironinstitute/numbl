function y = branch(x)
  % A function-scoped wildcard import: the bare call `leaf` must resolve to
  % impwild.leaf even though the import lives inside the function body.
  import impwild.*
  y = leaf(x) + 1;
end
