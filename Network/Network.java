package Network;

public class Network {
    Layer[] layers;
    HyperParameter params;
    NetworkData[] batchLearnData;

    public Network(HyperParameter network) {
        // Initialize number of layers to be one less than the
        // number of given layers
        // Input layer isn't considered as a layer for this model
        this.layers = new Layer[network.layerSizes.length - 1];

        // Initialize the layers in the network
        // Each value in the layers array corresponds
        // to the number of nodes in that layer of the network
        for (int i = 0; i < network.layerSizes.length - 1; i++) {
            this.layers[i] = new Layer(network.layerSizes[i], network.layerSizes[i + 1],
                    network.initialWeightsValues[i],
                    network.activation,
                    network.initialWeights,
                    network.costFunction);
        }

        params = network;
    }

    /**
     * 
     * @param trainingData
     * @param learnRate
     * @param regularization
     * @param momentum
     */
    public void learn(DataPoint[] trainingData, double learnRate, double regularization, double momentum) {
        if (batchLearnData == null || batchLearnData.length != trainingData.length) {
            batchLearnData = new NetworkData[trainingData.length];
            for (int i = 0; i < batchLearnData.length; i++) {
                batchLearnData[i] = new NetworkData(layers);
            }
        }

        for (int i = 0; i < trainingData.length; i++) {
            DataPoint point = trainingData[i];
            NetworkData batch = batchLearnData[i];
            new Thread(() -> updateGradients(point, batch)).start();
        }

        // Update weights and biases based on the calculated gradients
        for (int i = 0; i < layers.length; i++) {
            layers[i].gradient(learnRate / trainingData.length, regularization, momentum);
        }
    }

    /**
     * 
     * @param data
     * @param learnData
     */
    void updateGradients(DataPoint data, NetworkData learnData) {
        // Feed data through the network to calculate outputs.
        // Save all inputs/weightedinputs/activations along the way to use for
        // backpropagation.
        Double[] inputsToNextLayer = data.inputs;

        for (int i = 0; i < layers.length; i++) {
            inputsToNextLayer = layers[i].outputs(inputsToNextLayer, learnData.layerData[i]);
        }

        // -- Backpropagation --
        int outputLayerIndex = layers.length - 1;
        Layer outputLayer = layers[outputLayerIndex];
        LayerData outputLearnData = learnData.layerData[outputLayerIndex];

        // Update output layer gradients
        outputLayer.calculateOutputLayerNodeValues(outputLearnData, data.expectedOutputs);
        outputLayer.updateGradients(outputLearnData);

        // Update all hidden layer gradients
        for (int i = outputLayerIndex - 1; i >= 0; i--) {
            LayerData layerLearnData = learnData.layerData[i];
            Layer hiddenLayer = layers[i];

            hiddenLayer.calculateHiddenLayerNodeValues(layerLearnData, layers[i + 1],
                    learnData.layerData[i + 1].nodeValues);
            hiddenLayer.updateGradients(layerLearnData);
        }

    }

    /**
     * Gets the total cost of the currect network
     * 
     * @param inputs
     * @param expected
     * @return
     */
    public double cost(Double[][] inputs, double[][] expected) {
        double totalCost = 0;

        for (int i = 0; i < inputs.length; i++) {
            totalCost += loss(inputs[i], expected[i]);
        }

        return totalCost / inputs.length;
    }

    /**
     * Calculates the loss of the given input
     * 
     * @param <T>
     * @param inputs
     * @param expected
     * @return
     */
    public <T extends Number> Double loss(T[] inputs, double[] expected) {
        Double[] outputs = forwardPropogation(inputs);
        double cost = 0;

        for (int out = 0; out < outputs.length; out++) {
            cost += outCost(outputs[out], expected[out]);
        }

        return cost;
    }

    /**
     * Loops over every layer and returns the output layer's values
     * 
     * @param <T>
     * @param inputs
     * @return
     */
    @SuppressWarnings("unchecked")
    public <T extends Number> Double[] forwardPropogation(T[] inputs) {
        for (Layer l : layers) {
            inputs = (T[]) l.outputs(inputs);
        }
        return (Double[]) inputs;
    }

    /**
     * Error cost calculation
     * 
     * @param actual
     * @param expected
     * @return
     */
    public double outCost(double actual, double expected) {
        double error = actual - expected;
        return error * error;
    }

    /**
     * Classifies the input based on the output
     * 
     * @param inputs
     * @return
     */
    public Object[] classify(Double[] inputs) {
        Double[] outputs = forwardPropogation(inputs);
        int id = arrayMax(outputs);
        return new Object[] { id, outputs };
    }

    /**
     * Finds the max value in an array and returns the index of the value
     * 
     * @param inputs
     * @return
     */
    private int arrayMax(Double[] inputs) {
        int max = 0;
        for (int i = 0; i < inputs.length; i++) {
            if (inputs[i] > inputs[max]) {
                max = i;
            }
        }
        return max;
    }

    class NetworkData {
        public LayerData[] layerData;

        public NetworkData(Layer[] layers) {
            layerData = new LayerData[layers.length];
            for (int i = 0; i < layers.length; i++) {
                layerData[i] = new LayerData(layers[i]);
            }
        }
    }
}