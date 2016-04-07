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
- `inputs`   -- a vector of input values
- `∇inputs`  -- a vector of gradients for connected neurons

For our **WeightedNeuron** we will define the following additional fields:

- `weights`  -- a vector of learned weights
- `∇weights` -- a vector of gradients for the weights
- `bias`     -- the bias value
- `∇bias`    -- the gradient for the bias value

```julia
@neuron type WeightedNeuron <: Neuron
    value    :: Float32
    ∇        :: Float32

    inputs   :: DenseArray{Float32}
    ∇inputs  :: DenseArray{Float32}

    weights  :: DenseArray{Float32}
    ∇weights :: DenseArray{Float32}

    bias     :: DenseArray{Float32}
    ∇bias    :: DenseArray{Float32}
end
```

Next we define the forward computation for the neuron:
```julia
@neuron forward(neuron::WeightedNeuron) do
    # dot product of weights and inputs
    for i in 1:length(neuron.inputs)
        neuron.value += neuron.weights[i] * neuron.inputs[i]
    end
    # shift by the bias value
    neuron.value += neuron.bias
end
```

And finally we define the backward computation for the back-propogation algorithm:

```
@neuron backward(neuron::WeightedNeuron) do
    for i in 1:length(neuron.inputs[1])
        neuron.∇inputs[i] += neuron.weights[i] * neuron.∇
    end
    for i in 1:length(neuron.inputs[1])
        neuron.∇weights[i] += neuron.inputs[i] * neuron.∇
    end
    neuron.∇bias += neuron.∇
end
```

[1]: http://www.sciencedirect.com/science/article/pii/0893608089900208 "Multilayer feedforward networks are universal approximators. Hornik et al. 1989"
