public class Cost {
    public enum CostFunctionType {
        MeanSquaredError
    }

    public static CostFunction getCostFunction(CostFunctionType costFunctionType) {
        switch (costFunctionType) {
            case MeanSquaredError:
                return null;
            default:
                return null;
        }
    }

    static class MeanSquaredError implements CostFunction {

        public double cost(double[] predictedOutputs, double[] expectedOutputs) {
            // cost is sum (for all x,y pairs) of: 0.5 * (x-y)^2
            double cost = 0;
            for (int i = 0; i < predictedOutputs.length; i++) {
                double error = predictedOutputs[i] - expectedOutputs[i];
                cost += error * error;
            }
            return 0.5 * cost;
        }

        public double costDerivative(double predictedOutput, double expectedOutput) {
            return predictedOutput - expectedOutput;
        }
    }
}
