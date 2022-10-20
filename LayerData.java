public class LayerData {
    public double[] inputs;
    public double[] weightedInputs;
    public double[] activations;
    public double[] nodeValues;

    public LayerData(Layer layer) {
        weightedInputs = new double[layer.outNodes];
        activations = new double[layer.outNodes];
        nodeValues = new double[layer.outNodes];
    }
}