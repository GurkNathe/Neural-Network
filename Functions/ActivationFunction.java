public interface ActivationFunction {
    public double activationFunction(double[] inputs, int index);

    public double activationDerivative(double[] inputs, int index);
}