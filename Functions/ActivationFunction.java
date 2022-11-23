package Functions;
public interface ActivationFunction {
    public <T extends Number> double activationFunction(T[] inputs, int index);

    public <T extends Number> double activationDerivative(T[] inputs, int index);
}