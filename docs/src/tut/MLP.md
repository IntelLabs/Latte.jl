# Multi-Layer Perceptron

This tutorial is based on http://deeplearning.net/tutorial/mlp.html but uses Latte to implement the code examples.  

A Multi-Layer Perceptron can be described as a logistic regression classifier where the input is first transformed using a learnt non-linear transformation $\Phi$.  This intermediate layer performing the transformation is referred to as a **hidden layer**.  One hidden layer is sufficient to make MLPs a **universal approximator**[1][1].  

## The Model
An MLP with a single hidden layer can be represented graphically as follows:
![Single Hidden Layer MLP](http://deeplearning.net/tutorial/_images/mlp.png)

To understand this representation, we'll first define the properties of a singular neuron.  A **Neuron**, depicted in the figure as a circle, computes an output (typically called an activation) as a function of its inputs.  In this figure, an input is depicted as a directed edge flowing into a **Neuron**.  For an MLP, the function to compute the output of a **Neuron** begins with a weighted sum of the inputs.  Formally, if we describe the input as a vector of values $x$ and a vector of weights $w$, this operation can be written as $w \cdot x$ (dot product).  Next, we will shift this value by a learned bias $b$, then apply an activation function $s$ such as $tanh$, $sigmoid$ or $relu$.  The entire computation is written as $s(w \cdot x + b)$.  The values of $w$ and $b$ will be learnt using back-propogation (described later).

### Defining a Neuron in Latte
Defining a **Neuron** begins with a subtype of the abstract `Neuron` type.  We define the following fields:
- `value`   -- stores the output value of the neuron
- `inputs`  -- a vector of `Float32` input values
- `weights` -- a vector learned weights
- `bias`    -- the bias value

```julia
@neuron type WeightedNeuron <: Neuron
    value   :: Float32

    inputs  :: Vector{Float32}
    weights :: Vector{Float32}
    bias    :: Float32
end
```

Next we define the forward computation for the neuron.
```julia
@neuron forward(neuron::WeightedNeuron) do
    for i in 1:length(neuron.inputs)
        neuron.value += neuron.weights[i] * neuron.inputs[i]
    end
    neuron.value += neuron.bias
end
```


[1]: http://www.sciencedirect.com/science/article/pii/0893608089900208 "Multilayer feedforward networks are universal approximators. Hornik et al. 1989"
