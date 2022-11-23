package Network;

public class LayerData {
    public Double[] inputs;
    public Double[] weightedInputs;
    public Double[] activations;
    public Double[] nodeValues;

    public LayerData(Layer layer) {
        weightedInputs = new Double[layer.outNodes];
        activations = new Double[layer.outNodes];
        nodeValues = new Double[layer.outNodes];
    }
}