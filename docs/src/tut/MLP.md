# Multi-Layer Perceptron

This tutorial is based on [http://deeplearning.net/tutorial/mlp.html](http://deeplearning.net/tutorial/mlp.html) but uses Latte to implement the code examples.  

A Multi-Layer Perceptron can be described as a logistic regression classifier where the input is first transformed using a learnt non-linear transformation $\Phi$.  This intermediate layer performing the transformation is referred to as a **hidden layer**.  One hidden layer is sufficient to make MLPs a **universal approximator**[1][1].  

## The Model
An MLP with a single hidden layer can be represented graphically as follows:

![Single Hidden Layer MLP](http://deeplearning.net/tutorial/_images/mlp.png)

To understand this representation, we'll first define the properties of a singular neuron.  A **Neuron**, depicted in the figure as a circle, computes an output (typically called an activation) as a function of its inputs.  In this figure, an input is depicted as a directed edge flowing into a **Neuron**.  For an MLP, the function to compute the output of a **Neuron** begins with a weighted sum of the inputs.  Formally, if we describe the input as a vector of values $x$ and a vector of weights $w$, this operation can be written as $w \cdot x$ (dot product).  Next, we will shift this value by a learned bias $b$, then apply an activation function $s$ such as $tanh$, $sigmoid$ or $relu$.  The entire computation is written as $s(w \cdot x + b)$.  The values of $w$ and $b$ will be learnt using back-propogation.

## Defining a WeightedNeuron
Defining a **WeightedNeuron** begins with a subtype of the abstract `Neuron` type.  The `Neuron` type contains 4 default fields:
- `value`    -- contains the output value of the neuron
- `∇`        -- contains the gradient of the neuron
- `inputs`   -- a vector of vectors of input values.
- `∇inputs`  -- a vector of gradients for connected neurons

For our **WeightedNeuron** we will define the following additional fields:

- `weights`  -- a vector of learned weights
- `∇weights` -- a vector of gradients for the weights
- `bias`     -- the bias value
- `∇bias`    -- the gradient for the bias value

```julia
@neuron WeightedNeuron
    weights  :: DenseArray{Float32}
    ∇weights :: DenseArray{Float32}

    bias     :: DenseArray{Float32}
    ∇bias    :: DenseArray{Float32}
end
```

!!! note
    We do not define the default fields as using the @neuron macro for the type definition will specify them for us.  Furthermore the macro defines a constructor function that automatically initializes the default fields.

Next we define the forward computation for the neuron:
```julia
@neuron forward(neuron::WeightedNeuron) do
    # dot product of weights and inputs
    for i in 1:length(neuron.inputs[1])
        neuron.value += neuron.weights[i] * neuron.inputs[1][i]
    end
    # shift by the bias value
    neuron.value += neuron.bias[1]
end
```

And finally we define the backward computation for the back-propogation algorithm:

```julia
@neuron backward(neuron::WeightedNeuron) do
    for i in 1:length(neuron.inputs[1])
        neuron.∇inputs[1][i] += neuron.weights[i] * neuron.∇
    end
    for i in 1:length(neuron.inputs[1])
        neuron.∇weights[i] += neuron.inputs[1][i] * neuron.∇
    end
    neuron.∇bias[1] += neuron.∇
end
```

## Building a Layer using Ensembles and Connections
In Latte, a *layer* can be described as an `Ensemble` of `Neuron`s with a specific set of connections to one or more input `Ensemble`s.  To construct a **Hidden Layer** for our MLP, we will use an `Ensemble` of `WeightedNeurons` with each neuron connected to each all the neurons in the input `Ensemble`.

Our `FullyConnectedLayer` will be a Julia Function that instantiates an `Ensemble` of `WeightedNeuron`s and applies a full connection structure to the `input_ensemble`.  The signature looks like this:

```julia
function FCLayer(net::Net, input_ensemble::AbstractEnsemble, num_outputs::Int)
```

To construct a hidden layer with `num_neurons`, we begin by instantiating a 1-d `Array` to hold our `WeightedNeurons`.

```julia
    neurons = Array(WeightedNeuron, num_neurons)
```

Next we instantiate the parameters for our `WeightedNeurons`.  Note `xavier` refers to a function to initialize a random set of values using the *Xavier* (TODO: Reference) initialization scheme.  `xavier` and other initialization routines are provided as part of the Latte standard library.

```julia
    num_inputs = length(input_ensemble)
    weights = xavier(Float32, num_inputs, num_outputs)
    ∇weights = zeros(Float32, num_inputs, num_outputs)

    bias = zeros(Float32, 1, num_outputs)
    ∇bias = zeros(Float32, 1, num_outputs)
```

With our parameters initialized, we are ready to initialize our neurons.  Note that each `WeightedNeuron` instance uses a different row of parameter values.

```julia
    for i in 1:num_outputs
        neurons[i] = WeightedNeuron(view(weights, :, i), view(∇weights, :, i),
                                    view(bias, :, i), view(∇bias, :, i))
    end
```

Finally, we are ready to instantiate our Ensemble.

```
    ens = Ensemble(net, name, neurons, [Param(name,:weights, 1.0f0, 1.0f0), 
                                        Param(name,:bias, 2.0f0, 0.0f0)])
```

Then we add connections to each neuron in `input_ensemble`

```julia
    add_connections(net, input_ensemble, ens,
                    (i) -> (tuple([Colon() for d in size(input_ensemble)]... )))
```

Finally, we return the constructed Ensemble so it can be used as an input to another layer.
```
    return ens
end
```



[1]: http://www.sciencedirect.com/science/article/pii/0893608089900208 "Multilayer feedforward networks are universal approximators. Hornik et al. 1989"
