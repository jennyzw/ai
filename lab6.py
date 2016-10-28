# MIT 6.034 Lab 6: Neural Nets
# Written by Jessica Noss (jmn), Dylan Holmes (dxh), Jake Barnwell (jb16), and 6.034 staff
import math
import itertools
from nn_problems import *
from math import e
INF = float('inf')

#### NEURAL NETS ###############################################################

# Wiring a neural net

nn_half = [1]

nn_angle = [2,1]

nn_cross = [2,2,1]

nn_stripe = [3,1]

nn_hexagon = [6,1]

nn_grid = [4,2,1]

# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    return int(threshold <= x)

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    den = 1 + math.exp((-steepness)*(x-midpoint))
    return 1.0/den

def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    if x<0:
        return 0
    else:
        return x

# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return (-0.5)*((desired_output-actual_output)**2)

# Forward propagation

def node_value(node, input_values, neuron_outputs):  # STAFF PROVIDED
    """Given a node, a dictionary mapping input names to their values, and a
    dictionary mapping neuron names to their outputs, returns the output value
    of the node."""
    if isinstance(node, basestring):
        return input_values[node] if node in input_values else neuron_outputs[node]
    return node  # constant input, such as -1

def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""
    neuron_dict = {}
    #initialize the input nodes in dict w/ their output as their input
    for s in net.inputs:
        neuron_dict[s] = input_values.get(s,s)
    for neuron in net.topological_sort():
        output = 0
        for wire in net.get_wires(endNode=neuron):
            output += neuron_dict[wire.startNode]*wire.get_weight()
        neuron_dict[neuron] = threshold_fn(output)
    return (neuron_dict[net.get_output_neuron()], neuron_dict)

# Backward propagation warm-up
def gradient_ascent_step(func, inputs, step_size):
    """Given an unknown function of three variables and a list of three values
    representing the current inputs into the function, increments each variable
    by +/- step_size or 0, with the goal of maximizing the function output.
    After trying all possible variable assignments, returns a tuple containing:
    (1) the maximum function output found, and
    (2) the list of inputs that yielded the highest function output."""
    #raise NotImplementedError
    input1 = inputs[0]
    input2 = inputs[1]
    input3 = inputs[2]
    highest_output = -INF
    best_list = inputs
    i1=(input1,input1+step_size,input1-step_size)
    i2=(input2,input2+step_size,input2-step_size)
    i3=(input3,input3+step_size,input3-step_size)
    combinations = list(itertools.product(i1,i2,i3))
    for combo in combinations:
        a=combo[0]
        b=combo[1]
        c=combo[2]
        if func(a,b,c) > highest_output:
            highest_output = func(a,b,c)
            best_list = [a,b,c]
    return (highest_output, best_list)

def get_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons, and
    Wires whose outputs/values are required to update this wire's weight."""
    #recursion, chnage the wire each time, keep net same. add to unordered set
    if(wire.endNode == net.get_output_neuron()):
        dependencies = set([wire.startNode, wire.endNode, wire])
    #updating the weight requires:
    #output from node A
    #current weight of the wire from A to B
    #output of node B
    #all neurons and weights downstream to the final layer
    for w in net.get_wires(startNode=wire.startNode):
        dependencies.add(w.startNode)
        dependencies.add(w.endNode)
        dependencies.add(w)
    return dependencies

# Backward propagation
def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net. Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """
    #raise NotImplementedError
    delta_values = {}
    neurons = reversed(net.topological_sort())
    for neuron in neurons:
        output = neuron_outputs[neuron]
        if net.is_output_neuron(neuron):
            delta_values[neuron] = output*(1-output)*(desired_output-output)
        else:
            wsum=0
            for wire in net.get_wires(neuron):
                wsum += wire.get_weight()*delta_values[wire.endNode]
            delta_values[neuron] = output*(1-output)*wsum
    return delta_values

def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""
    delta_values = calculate_deltas(net, desired_output, neuron_outputs)
    for wire in net.get_wires():
        wire.weight += r*node_value(wire.startNode, input_values, neuron_outputs)*delta_values[wire.endNode]
    return net

def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""
    #raise NotImplementedError
    count = 0
    (actual_output, neuron_outputs) = forward_prop(net, input_values, sigmoid)
    while accuracy(desired_output, actual_output) <= minimum_accuracy:
        net = update_weights(net, input_values, desired_output, neuron_outputs, r)
        count += 1
        (actual_output, neuron_outputs) = forward_prop(net, input_values, sigmoid) #one step of forward prop
    return (net, count)

# Training a neural net

ANSWER_1 = None
ANSWER_2 = None
ANSWER_3 = None
ANSWER_4 = None
ANSWER_5 = None

ANSWER_6 = None
ANSWER_7 = None
ANSWER_8 = None
ANSWER_9 = None

ANSWER_10 = "D"
ANSWER_11 = None
ANSWER_12 = None


#### SURVEY ####################################################################

NAME = None
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
