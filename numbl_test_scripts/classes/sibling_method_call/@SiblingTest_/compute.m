function out = compute(obj, x)
  % Calls sibling method helper_work with obj as 3rd arg (not first)
  % This is the chebfun pattern: sampleTest(op, values, f, data, pref)
  % where f is the class instance but not the first argument.
  out = helper_work(x, 10, obj);
end
