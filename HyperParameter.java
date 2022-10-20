public class HyperParameter {
    Activation.ActivationFunctionType activation;
    InitialWeights.InitialWeightsType initialWeights;
    Cost.CostFunctionType costFunction;
    double[][] initialWeightsValues;
    int[] layerSizes;
    double initialLearningRate;
    double learnRateDecay;
    double momentum;
    double regularization;

    public HyperParameter(Activation.ActivationFunctionType activation,
            InitialWeights.InitialWeightsType initialWeights, Cost.CostFunctionType costFunction,
            double[][] initialWeightsValues, int[] layerSizes,
            double initialLearningRate,
            double learnRateDecay,
            double momentum,
            double regularization) {
        this.activation = activation;
        this.initialWeights = initialWeights;
        this.costFunction = costFunction;
        this.initialWeightsValues = initialWeightsValues;
        this.layerSizes = layerSizes;
        this.initialLearningRate = initialLearningRate;
        this.learnRateDecay = learnRateDecay;
        this.momentum = momentum;
        this.regularization = regularization;
    }
}
