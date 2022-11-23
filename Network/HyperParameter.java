package Network;

import Functions.Activation;
import Functions.Cost;
import Functions.InitialWeights;

public class HyperParameter {
    Activation.ActivationFunctionType activation;
    InitialWeights.InitialWeightsType initialWeights;
    Cost.CostFunctionType costFunction;
    double[][] initialWeightsValues;
    int[] layerSizes;
    Double[] learnParams;

    public HyperParameter(Activation.ActivationFunctionType activation,
            InitialWeights.InitialWeightsType initialWeights, Cost.CostFunctionType costFunction,
            double[][] initialWeightsValues, int[] layerSizes,
            Double initialLearningRate,
            Double learnRateDecay,
            Double momentum,
            Double regularization,
            Double... learnParams) {
        this.activation = activation;
        this.initialWeights = initialWeights;
        this.costFunction = costFunction;
        this.initialWeightsValues = initialWeightsValues;
        this.layerSizes = layerSizes;
        this.learnParams = learnParams.length != 4 ? new Double[] { 0.05, 0.075, 0.9, 0.1 } : learnParams;
    }
}
