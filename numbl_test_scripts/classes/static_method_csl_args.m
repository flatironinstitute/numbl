function static_method_csl_args()
% Test that cell expansion (CSL) works in static method call arguments.
% e.g. ClassName.method(cell{:})

    obj = MyHelper();
    c = {obj, 'hello'};
    [a, b] = MyHelper.grab_first_two(c{:});
    assert(isa(a, 'MyHelper'));
    assert(strcmp(b, 'hello'));

    disp('SUCCESS');
end
