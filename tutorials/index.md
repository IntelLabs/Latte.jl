---
title: Tutorials
layout: page
---

## Language
Latte is a domain specific language designed for expressing deep neural
networks in a natural way.  Here we will cover the language primitives that can
then be used to construct deep neural architectures.  Using Latte begins with an
instance of the `Net` type.  Users then adds `Ensemble`s of `Neuron`s to the `Net`
and apply `Connection`s between them.

### Neuron
The `Neuron` is the fundamental unit of a neural network.  They take in any number
of inputs and compute an output value called an `activation`.  In Latte, a neuron
can compute its output as any arbitrary function of its inputs.  The most common
`Neuron` type used in modern deep neural networks is the `WeightedNeuron`.  A
`WeightedNeuron` computes its output activation as a weighted sum of its inputs.
Another example of a `Neuron` is a `MaxNeuron` that computes its output as the
maximum of its inputs.

### Ensemble
TODO

### Connection
TODO
