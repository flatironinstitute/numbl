% Arity errors for output, higher-order, file I/O, filesystem, web, path,
% environment, numerics, and graphics builtins (MATLAB's messages and
% identifiers).

f = [tempname '.txt'];
fid = fopen(f, 'w');
fprintf(fid, '1 2 3\n');
fclose(fid);
fid = fopen(f, 'r');

% ── output and higher-order ─────────────────────────────────────────────
check_err('disp()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('disp(1, 2)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('fprintf()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('toc(1, 2)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('[a, b] = toc', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('cellfun(@sin)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('arrayfun(@sin)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('structfun(@sin)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('feval()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('bsxfun(@plus, 1)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('bsxfun(@plus, 1, 2, 3)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('quadgk(@sin, 0)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('subsref(1)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('builtin()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('onCleanup(1, 2)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
[a, b] = builtin('size', ones(2, 3));
assert(a == 2 && b == 3);

% ── file I/O ────────────────────────────────────────────────────────────
check_err('fopen()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('[a, b, c, d, e] = fopen(fid)', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('fclose()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('fclose(1, 2)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('fgetl(fid, 2)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('fgets(fid, 2, 3)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('fscanf(fid)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('feof()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('ferror()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('fread()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('[a, b, c] = fread(fid)', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('fwrite(fid)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('fseek(fid, 0)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('ftell(fid, 2)', 'MATLAB:maxrhs', 'Too many input arguments.');
% the arity checks did not consume any input
assert(isequal(fscanf(fid, '%d'), [1; 2; 3]));
fclose(fid);

% ── filesystem and web ──────────────────────────────────────────────────
check_err('mkdir()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('mkdir(''a'', ''b'', ''c'')', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('[a, b, c, d] = mkdir(tempdir)', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('rmdir()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('movefile()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('copyfile()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('fileattrib(''a'', ''+w'', ''u'', ''x'', ''y'')', 'MATLAB:maxrhs', ...
  'Too many input arguments.');
check_err('[a, b, c, d] = fileattrib(f)', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('dir(''.'', 2)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('tempdir(1)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('tempname(1, 2)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('x = delete(f)', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('websave(''a'')', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('webread()', 'MATLAB:minrhs', 'Not enough input arguments.');
delete(f);

% ── path and environment ────────────────────────────────────────────────
check_err('mfilename(1, 2)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('path(1, 2, 3)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('getenv(''a'', ''b'')', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('maxNumCompThreads(1, 2)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('setenv(''a'', ''b'', ''c'')', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('x = setenv(''a'', ''b'')', 'MATLAB:TooManyOutputs', 'Too many output arguments.');
check_err('pwd(1)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('cd(1, 2)', 'MATLAB:maxrhs', 'Too many input arguments.');

% ── numerics ────────────────────────────────────────────────────────────
check_err('ode45(@(t, y) y, [0 1])', 'MATLAB:odearguments:TspanOrY0NotSupplied', ...
  'When the first argument to ode45 is a function handle, the tspan and y0 arguments must be supplied.');
check_err('deval(1)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('deval(1, 2, 3, 4)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('gmres(1)', 'MATLAB:gmres:NumInputs', 'Not enough input arguments.');

% ── graphics ────────────────────────────────────────────────────────────
check_err('x = drawnow', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('set()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('r = hold', 'MATLAB:TooManyOutputs', 'Too many output arguments.');
check_err('r = grid', 'MATLAB:TooManyOutputs', 'Too many output arguments.');
check_err('r = box', 'MATLAB:TooManyOutputs', 'Too many output arguments.');
check_err('r = shading', 'MATLAB:TooManyOutputs', 'Too many output arguments.');
check_err('shading(1, 2, 3)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('colormap(1, 2, 3)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('[a, b] = colormap', 'MATLAB:TooManyOutputs', 'Too many output arguments.');
check_err('[a, b, c, d] = contourf(magic(4))', 'MATLAB:TooManyOutputs', ...
  'Too many output arguments.');
check_err('[a, b, c] = contour(magic(4))', 'MATLAB:TooManyOutputs', ...
  'Too many output arguments.');
check_err('[a, b] = pcolor(magic(3))', 'MATLAB:TooManyOutputs', 'Too many output arguments.');
check_err('[a, b, c] = title(''x'')', 'MATLAB:TooManyOutputs', 'Too many output arguments.');
check_err('[a, b] = colorbar', 'MATLAB:TooManyOutputs', 'Too many output arguments.');
% cla returns the axes when an output is requested
ax = cla; %#ok<NASGU>
close all;

disp('SUCCESS')

function check_err(stmt, id, msg)
  ok = false;
  fid = evalin('caller', 'fid'); %#ok<NASGU> used by eval'd statements
  f = evalin('caller', 'f'); %#ok<NASGU>
  try
    eval([stmt ';']);
  catch e
    ok = true;
    assert(strcmp(e.identifier, id), ...
      sprintf('%s: identifier ''%s'', expected ''%s''', stmt, e.identifier, id));
    assert(strcmp(e.message, msg), ...
      sprintf('%s: message ''%s'', expected ''%s''', stmt, e.message, msg));
  end
  assert(ok, sprintf('%s: expected an error', stmt));
end
