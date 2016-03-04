export RNNBlock

function RNNBlock(name::Symbol, net::Net, input::AbstractEnsemble)
    @assert ndims(input) == 1 "RNNLayer currently only supports 1-d inputs"
    x = InnerProductLayer(symbol(name, :x), net, input, length(input))
    state = FullyConnectedEnsemble(symbol(name, :state), net, length(input),
                                   length(input))
    # h = AddLayer(symbol(name, :tanh), net, x, state)
    sum = AddLayer(symbol(name, :sum), net, x, state)
    h = TanhLayer(symbol(name, :tanh), net, sum)
    add_connections(net, h, state,
                    (i) -> (tuple([Colon() for d in size(state)]... ));
                    recurrent=true)
    InnerProductLayer(name, net, h, length(input))
end
