% Test that class methods take priority over constructors with the same name,
% even when the argument type is Unknown at compile time.
% When calling Info(obj) where obj is an Animal (but the compiler doesn't
% know that), MATLAB should dispatch to @Animal/Info (the method), not
% @Info/Info (the constructor).

% Create via helper function so the return type is Unknown at compile time.
a = make_animal('cat');

% Info(a) should call @Animal/Info, not @Info constructor
result = Info(a);

assert(strcmp(result, 'Animal: cat'));

disp('SUCCESS');
