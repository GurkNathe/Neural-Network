package Network;

import Functions.*;

public class Layer {
    Double[] weights, weightGradient, weightVelocities;
    Double[] biases, biasesGradient, biasesVelocities;

    InitialWeightsFunction init;
    ActivationFunction activation;
    CostFunction cost;

    // Number of input nodes and output nodes respectively
    public int inNodes, outNodes;

    /**
     * Initializes a layer in a neural network
     * 
     * @param inputNodes
     * @param outputNodes
     * @param weightsInit
     * @param activate
     * @param initialWeights
     * @param costFunc
     */
    public Layer(int inputNodes, int outputNodes, double[] weightsInit, Activation.ActivationFunctionType activate,
            InitialWeights.InitialWeightsType initialWeights, Cost.CostFunctionType costFunc) {
        // Initialize weights and biases arrays
        weights = new Double[inputNodes * outputNodes];
        biases = new Double[outputNodes];

        // Initialize gradient arrays
        weightGradient = new Double[inputNodes * outputNodes];
        biasesGradient = new Double[outputNodes];

        // Initialize velocity arrays
        weightVelocities = new Double[inputNodes * outputNodes];
        biasesVelocities = new Double[outputNodes];

        inNodes = inputNodes;
        outNodes = outputNodes;

        // If no weight initialization function given, Xavier function is used
        init = InitialWeights.getInitialWeightsFunction(initialWeights);

        // If no activation function given, use sigmoid activation function as default
        activation = Activation.getActivationFunction(activate);

        cost = Cost.getCostFunction(costFunc);

        initializeWeights(weightsInit);
    }

    /**
     * Handles gradient descent
     * 
     * @param learnRate
     * @param regularization
     * @param momentum
     */
    public void gradient(double learnRate, double regularization, double momentum) {
        double weightDecay = 1 - regularization * learnRate;

        for (int i = 0; i < weights.length; i++) {
            double weight = weights[i];
            double velocity = weightVelocities[i] * momentum - weightGradient[i] * learnRate;
            weightVelocities[i] = velocity;
            weights[i] = weight * weightDecay + velocity;
            weightGradient[i] = 0.0;
        }

        for (int i = 0; i < outNodes; i++) {
            double velocity = biasesVelocities[i] * momentum - biasesGradient[i] * learnRate;
            biasesVelocities[i] = velocity;
            biases[i] += velocity;
            biasesGradient[i] = 0.0;
        }
    }

    // Initializes the weights of the layer based on the given initialization
    // function
    private void initializeWeights(double[] initVals) {
        for (int i = 0; i < inNodes; i++) {
            for (int j = 0; j < outNodes; j++) {
                setWeight(i, j, init.getInitialWeight(initVals));
            }
        }
    }

    // Calculates the output values fed through the current layer
    public <T extends Number> Double[] outputs(T[] inputs) {
        Double[] output = new Double[outNodes];

        for (int out = 0; out < outNodes; out++) {
            double weightedInput = biases[out];

            for (int in = 0; in < inNodes; in++) {
                weightedInput += (Double) inputs[in] * getWeight(in, out);
            }

            output[out] = weightedInput;
        }

        // Apply activation function
        for (int outNode = 0; outNode < outNodes; outNode++) {
            output[outNode] = activation.activationFunction(output, outNode);
        }

        return output;
    }

    public <T extends Number> Double[] outputs(T[] inputs, LayerData learnData) {
        learnData.inputs = (Double[]) inputs;

        for (int nodeOut = 0; nodeOut < outNodes; nodeOut++) {
            double weightedInput = biases[nodeOut];

            for (int nodeIn = 0; nodeIn < inNodes; nodeIn++) {
                weightedInput += (Double) inputs[nodeIn] * getWeight(nodeIn, nodeOut);
            }

            learnData.weightedInputs[nodeOut] = weightedInput;
        }

        // Apply activation function
        for (int i = 0; i < learnData.activations.length; i++) {
            learnData.activations[i] = activation.activationFunction(learnData.weightedInputs, i);
        }

        return learnData.activations;
    }

    public void calculateOutputLayerNodeValues(LayerData layerLearnData, Double[] expectedOutputs) {
        for (int i = 0; i < layerLearnData.nodeValues.length; i++) {
            // Evaluate partial derivatives for current node: cost/activation &
            // activation/weightedInput
            double costDerivative = cost.costDerivative(layerLearnData.activations[i], expectedOutputs[i]);
            double activationDerivative = activation.activationDerivative(layerLearnData.weightedInputs, i);
            layerLearnData.nodeValues[i] = costDerivative * activationDerivative;
        }
    }

    public void calculateHiddenLayerNodeValues(LayerData layerLearnData, Layer oldLayer, Double[] oldNodeValues) {
        for (int newNodeIndex = 0; newNodeIndex < outNodes; newNodeIndex++) {
            double newNodeValue = 0;
            for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.length; oldNodeIndex++) {
                // Partial derivative of the weighted input with respect to the input
                double weightedInputDerivative = oldLayer.getWeight(newNodeIndex, oldNodeIndex);
                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
            }
            newNodeValue *= activation.activationDerivative(layerLearnData.weightedInputs, newNodeIndex);
            layerLearnData.nodeValues[newNodeIndex] = newNodeValue;
        }
    }

    public void updateGradients(LayerData layerLearnData) {
        for (int nodeOut = 0; nodeOut < outNodes; nodeOut++) {
            double nodeValue = layerLearnData.nodeValues[nodeOut];
            for (int nodeIn = 0; nodeIn < inNodes; nodeIn++) {
                // Evaluate the partial derivative: cost / weight of current connection
                double derivativeCostWrtWeight = layerLearnData.inputs[nodeIn] * nodeValue;
                // The costGradientW array stores these partial derivatives for each weight.
                // Note: the derivative is being added to the array here because ultimately we
                // want
                // to calculate the average gradient across all the data in the training batch
                weightGradient[getIndex(nodeIn, nodeOut)] += derivativeCostWrtWeight;
            }
        }

        // Update cost gradient with respect to biases (lock for multithreading)
        for (int nodeOut = 0; nodeOut < outNodes; nodeOut++) {
            // Evaluate partial derivative: cost / bias
            double derivativeCostWrtBias = 1 * layerLearnData.nodeValues[nodeOut];
            biasesGradient[nodeOut] += derivativeCostWrtBias;
        }
    }

    public int getIndex(int inputNeuronIndex, int outputNeuronIndex) {
        return outputNeuronIndex * inNodes + inputNeuronIndex;
    }

    /**
     * Getter for weight array
     * 
     * Example: weight[input][output]
     * 
     * @param input  : index of first layer of weights
     * @param output : index of weight in the layer
     * @return value in array at index
     */
    public Double getWeight(int input, int output) {
        return weights[input * output];
    }

    /**
     * Getter for weightGradient array
     * 
     * Example: weight[input][output]
     * 
     * @param input  : index of first layer of weights
     * @param output : index of weight in the layer
     * @return value in array at index
     */
    public double getWeightGradient(int input, int output) {
        return weightGradient[input * output];
    }

    /**
     * Getter for weightVelocities array
     * 
     * Example: weight[input][output]
     * 
     * @param input  : index of first layer of weights
     * @param output : index of weight in the layer
     * @return value in array at index
     */
    public double getWeightVelocity(int input, int output) {
        return weightVelocities[input * output];
    }

    /**
     * Setter for weight array
     * 
     * Example: weight[input][output] = value
     * 
     * @param input  : index of first layer of weights
     * @param output : index of weight in the layer
     * @param value  : value to be used to modify weight value
     * @return value in array at index
     */
    public void setWeight(int input, int output, double value) {
        weights[input * output] = value;
    }

    /**
     * Setter for weight array
     * 
     * Example: weight[input][output] = value
     * 
     * @param input  : index of first layer of weights
     * @param output : index of weight in the layer
     * @param value  : value to be used to modify weight value
     * @return value in array at index
     */
    public void setWeightGradient(int input, int output, double value) {
        weightGradient[input * output] = value;
    }

    /**
     * Setter for weight array
     * 
     * Example: weight[input][output] = value
     * 
     * @param input  : index of first layer of weights
     * @param output : index of weight in the layer
     * @param value  : value to be used to modify weight value
     * @return value in array at index
     */
    public void setWeightVelocity(int input, int output, double value) {
        weightVelocities[input * output] = value;
    }

    /**
     * Setter for weight array
     * 
     * Example: weight[input][output] += value
     * 
     * @param input  : index of first layer of weights
     * @param output : index of weight in the layer
     * @param value  : value to be used to modify weight value
     * @return value in array at index
     */
    public void addWeight(int input, int output, double value) {
        weights[input * output] += value;
    }

    /**
     * Setter for weight array
     * 
     * Example: weight[input][output] += value
     * 
     * @param input  : index of first layer of weights
     * @param output : index of weight in the layer
     * @param value  : value to be used to modify weight value
     * @return value in array at index
     */
    public void addWeightGradient(int input, int output, double value) {
        weightGradient[input * output] += value;
    }

    /**
     * Setter for weight array
     * 
     * Example: weight[input][output] += value
     * 
     * @param input  : index of first layer of weights
     * @param output : index of weight in the layer
     * @param value  : value to be used to modify weight value
     * @return value in array at index
     */
    public void addWeightVelocity(int input, int output, double value) {
        weightVelocities[input * output] += value;
    }

    /**
     * Setter for weight array
     * 
     * Example: weight[input][output] *= value
     * 
     * @param input  : index of first layer of weights
     * @param output : index of weight in the layer
     * @param value  : value to be used to modify weight value
     * @return value in array at index
     */
    public void multiplyWeight(int input, int output, double value) {
        weights[input * output] *= value;
    }

    /**
     * Setter for weight array
     * 
     * Example: weight[input][output] *= value
     * 
     * @param input  : index of first layer of weights
     * @param output : index of weight in the layer
     * @param value  : value to be used to modify weight value
     * @return value in array at index
     */
    public void multiplyWeightGradient(int input, int output, double value) {
        weightGradient[input * output] *= value;
    }

    /**
     * Setter for weight array
     * 
     * Example: weight[input][output] *= value
     * 
     * @param input  : index of first layer of weights
     * @param output : index of weight in the layer
     * @param value  : value to be used to modify weight value
     * @return value in array at index
     */
    public void multiplyWeightVelocity(int input, int output, double value) {
        weightVelocities[input * output] *= value;
    }

    /**
     * Helper method for initializing weights matrix.
     */
    public static double[][] initializeWeights(int[] layers) {
        double[][] layerWeights = new double[layers.length - 1][2];

        for (int i = 0; i < layerWeights.length; i++) {
            for (int j = 0; j < layerWeights[i].length; j++) {
                layerWeights[i][j] = layers[i + j];
            }
        }

        return layerWeights;
    }
}