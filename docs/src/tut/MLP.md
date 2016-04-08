# Multi-Layer Perceptron

This tutorial is based on
[http://deeplearning.net/tutorial/mlp.html](http://deeplearning.net/tutorial/mlp.html)
but uses Latte to implement the code examples.  

A Multi-Layer Perceptron can be described as a logistic regression classifier
where the input is first transformed using a learnt non-linear transformation
$\Phi$.  This intermediate layer performing the transformation is referred to
as a **hidden layer**.  One hidden layer is sufficient to make MLPs a
**universal approximator** [[1][1]].  

## Stochastic Gradient Descent
Gradient Descent is a simple algorithm that repeatedly makes small steps
downward on an error surface defined by a loss function of some parameters.
Traditional Gradient Descent computes the gradient after a pass over the
entire input data set.  In practice, Gradient Descent proceeds more quickly
when the gradient is estimated from just a few examples at time.  This extreme form
computes the gradient for a single training input at a time:
```julia
for (input, expected_result) in training_set:
    loss = f(params, input, expected_result)
    ∇_loss_wrt_params = ...  # compute gradients
    params -= learning_rate * ∇_loss_wrt_params
    if stop_condition_met()
        return params
    end
end
```

In practice, SGD for deep learning is performed using *minibatches*, where a
`minibatch` of inputs are used to estimate the gradients.  This technique
typically reduces variance in the estimation of the gradient, but more
importantly allows implementations to make better use of the memory hierarchy
in modern computers.

## The Model
An MLP with a single hidden layer can be represented graphically as follows:

![Single Hidden Layer MLP](http://deeplearning.net/tutorial/_images/mlp.png)

To understand this representation, we'll first define the properties of a
singular neuron.  A **Neuron**, depicted in the figure as a circle, computes an
output (typically called an activation) as a function of its inputs.  In this
figure, an input is depicted as a directed edge flowing into a **Neuron**.  For
an MLP, the function to compute the output of a **Neuron** begins with a
weighted sum of the inputs.  Formally, if we describe the input as a vector of
values $x$ and a vector of weights $w$, this operation can be written as $w
\cdot x$ (dot product).  Next, we will shift this value by a learned bias $b$,
then apply an activation function $s$ such as $tanh$, $sigmoid$ or $relu$.  The
entire computation is written as $s(w \cdot x + b)$.

TODO: Discuss back-prop and tie into SGD

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
@neuron type WeightedNeuron
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
function FCLayer(name::Symbol, net::Net, input_ensemble::AbstractEnsemble, num_neurons::Int)
```

To construct a hidden layer with `num_neurons`, we begin by instantiating a 1-d `Array` to hold our `WeightedNeurons`.

```julia
    neurons = Array(WeightedNeuron, num_neurons)
```

Next we instantiate the parameters for our `WeightedNeurons`.  Note `xavier` refers to a function to initialize a random set of values using the *Xavier* (TODO: Reference) initialization scheme.  `xavier` and other initialization routines are provided as part of the Latte standard library.

```julia
    num_inputs = length(input_ensemble)
    weights = xavier(Float32, num_inputs, num_neurons)
    ∇weights = zeros(Float32, num_inputs, num_neurons)

    bias = zeros(Float32, 1, num_neurons)
    ∇bias = zeros(Float32, 1, num_neurons)
```

With our parameters initialized, we are ready to initialize our neurons.  Note that each `WeightedNeuron` instance uses a different row of parameter values.

```julia
    for i in 1:num_neurons
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

## Constructing an MLP using Net
To construct an MLP we instantiate the `Net` type with a batch size of $100$.  Then we use the Latte standard library provided `HDF5DataLayer` that constructs an input ensemble that reads from `HDF5` datasets.  (TODO: Link to explanation of Latte's HDF5 format).  Then we construct two `FCLayer`s using the function that we defined.  Finally we use two more Latte standard library layers as output layers.  The `SoftmaxLoss` layer is used for train the network and the `AccuracyLayer` is used for test the network.
```julia
using Latte

net = Net(100)
data, label = HDF5DataLayer(net, "data/train.txt", "data/test.txt")

fc1 = FCLayer(:fc1, net, data, 100)
fc2 = FCLayer(:fc2, net, fc1, 10)

loss     = SoftmaxLossLayer(:loss, net, fc2, label)
accuracy = AccuracyLayer(:accuracy, net, fc2, label)

params = SolverParameters(
    lr_policy    = LRPolicy.Inv(0.01, 0.0001, 0.75),
    mom_policy   = MomPolicy.Fixed(0.9),
    max_epoch    = 50,
    regu_coef    = .0005)
sgd = SGD(params)
solve(sgd, net)
```

## Training
We will train the above MLP on the MNIST digit recognition dataset.  For your convenience the code in this tutorial has been provided in `tutorials/mlp/mlp.jl`.  Note that the name `WeightedNeuron` was replaced with `MLPNeuron` to resolve conflicts with the existing `WeightedNeuron` definition in the Latte standard library.  To train the network, first download and convert the dataset by running `tutorials/mlp/data/get-data.sh`.  Then train by running the script `julia mlp.jl`.  You should the following output that shows the loss values and test results:
```
...
INFO: 07-Apr 15:15:22 - Entering solve loop
INFO: 07-Apr 15:15:23 - Iter 20 - Loss: 1.4688001
INFO: 07-Apr 15:15:24 - Iter 40 - Loss: 0.6913204
INFO: 07-Apr 15:15:25 - Iter 60 - Loss: 0.6053091
INFO: 07-Apr 15:15:26 - Iter 80 - Loss: 0.6043377
INFO: 07-Apr 15:15:27 - Iter 100 - Loss: 0.57204634
INFO: 07-Apr 15:15:28 - Iter 120 - Loss: 0.500179
INFO: 07-Apr 15:15:28 - Iter 140 - Loss: 0.40663132
INFO: 07-Apr 15:15:29 - Iter 160 - Loss: 0.3704785
INFO: 07-Apr 15:15:29 - Iter 180 - Loss: 0.3620596
INFO: 07-Apr 15:15:30 - Iter 200 - Loss: 0.46897307
INFO: 07-Apr 15:15:30 - Iter 220 - Loss: 0.45075363
INFO: 07-Apr 15:15:31 - Iter 240 - Loss: 0.3376474
INFO: 07-Apr 15:15:31 - Iter 260 - Loss: 0.5301368
INFO: 07-Apr 15:15:32 - Iter 280 - Loss: 0.28490248
INFO: 07-Apr 15:15:32 - Iter 300 - Loss: 0.33110633
INFO: 07-Apr 15:15:33 - Iter 320 - Loss: 0.26910272
INFO: 07-Apr 15:15:33 - Iter 340 - Loss: 0.32226878
INFO: 07-Apr 15:15:33 - Iter 360 - Loss: 0.3838666
INFO: 07-Apr 15:15:34 - Iter 380 - Loss: 0.24588501
INFO: 07-Apr 15:15:34 - Iter 400 - Loss: 0.4209111
INFO: 07-Apr 15:15:35 - Iter 420 - Loss: 0.25582874
INFO: 07-Apr 15:15:35 - Iter 440 - Loss: 0.3958639
INFO: 07-Apr 15:15:36 - Iter 460 - Loss: 0.27812394
INFO: 07-Apr 15:15:36 - Iter 480 - Loss: 0.45379284
INFO: 07-Apr 15:15:37 - Iter 500 - Loss: 0.35272872
INFO: 07-Apr 15:15:38 - Iter 520 - Loss: 0.39787623
INFO: 07-Apr 15:15:38 - Iter 540 - Loss: 0.30763283
INFO: 07-Apr 15:15:39 - Iter 560 - Loss: 0.35435736
INFO: 07-Apr 15:15:40 - Iter 580 - Loss: 0.33140996
INFO: 07-Apr 15:15:41 - Iter 600 - Loss: 0.34410283
INFO: 07-Apr 15:15:41 - Epoch 1 - Testing...
INFO: 07-Apr 15:15:44 - Epoch 1 - Test Result: 90.88118%
...
```

[1]: http://www.sciencedirect.com/science/article/pii/0893608089900208 "Multilayer feedforward networks are universal approximators. Hornik et al. 1989"

