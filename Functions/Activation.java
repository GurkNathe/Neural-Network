package Functions;

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
        public <T extends Number> double activationFunction(T[] inputs, int index) {
            return 1 / (1 + Math.exp(-inputs[index].doubleValue()));
        }

        @Override
        public <T extends Number> double activationDerivative(T[] inputs, int index) {
            double a = activationFunction(inputs, index);
            return a * (1 - a);
        }
    }
}
