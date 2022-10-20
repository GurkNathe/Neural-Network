public class Activation {
    public enum ActivationFunctionType {
        Sigmoid
    }

    public static ActivationFunction getActivationFunction(ActivationFunctionType activationFunction) {
        switch (activationFunction) {
            case Sigmoid:
                return new Sigmoid();
            default:
                return new Sigmoid();
        }
    }

    static class Sigmoid implements ActivationFunction {
        public double activationFunction(double[] inputs, int index) {
            return 1 / (1 + Math.exp(-inputs[index]));
        }

        public double activationDerivative(double[] inputs, int index) {
            double a = activationFunction(inputs, index);
            return a * (1 - a);
        }
    }
}
