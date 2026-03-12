function r = chain_middle(x)
    r = chain_inner(x) + chain_inner(x + 1);
end
