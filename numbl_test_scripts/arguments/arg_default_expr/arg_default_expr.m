% Test: arguments-block default expressions that reference earlier arguments
% (and call helper/static functions), plus name-value struct defaults.
% Verified against MATLAB R2025b.

% --- Plain function: default of b is an expression referencing a ---
assert(argdefs(5) == 105, 'b default makeb(5) = 105');
assert(argdefs(5, 7) == 7, 'explicit b = 7');
assert(argdefs() == 100, 'a default 0 -> b default makeb(0) = 100');

% --- Name-value defaults ---
assert(argnv(2) == 12, 'opts.k default 10, 2 + 10 = 12');
assert(argnv(2, 'k', 100) == 102, 'opts.k = 100');

% --- Class constructor: output var is prepended to params, so the
%     arguments block is offset by one; the default for mi calls a static
%     method on the earlier arg a. ---
o1 = ArgCtor(7);
assert(o1.a == 7, 'a = 7');
assert(o1.mi == 1007, 'mi default = defaultMi(7) = 1007');
assert(strcmp(o1.method, 'DtN'), 'method default DtN');

o2 = ArgCtor(7, 99);
assert(o2.mi == 99, 'explicit mi = 99');

o3 = ArgCtor(7, 99, 'method', 'ItI');
assert(strcmp(o3.method, 'ItI'), 'method = ItI');

disp('SUCCESS')

function out = argdefs(a, b)
  arguments
    a = 0
    b = makeb(a)
  end
  out = b;
end

function v = makeb(a)
  v = a + 100;
end

function out = argnv(a, opts)
  arguments
    a
    opts.k = 10
  end
  out = a + opts.k;
end
