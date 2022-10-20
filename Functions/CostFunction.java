public interface CostFunction {
    public double cost(double[] predictedOutputs, double[] expectedOutputs);

    public double costDerivative(double predictedOutput, double expectedOutput);
}