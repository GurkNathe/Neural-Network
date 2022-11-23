package Network;

public class DataPoint {
    public Double[] inputs;
    public Double[] expectedOutputs;
    public int label;

    public DataPoint(Double[] inputs, int label, int numLabels) {
        this.inputs = inputs;
        this.label = label;
        expectedOutputs = CreateOneHot(label, numLabels);
    }

    public static Double[] CreateOneHot(int index, int num) {
        Double[] oneHot = new Double[num];
        oneHot[index] = 1.0;
        return oneHot;
    }
}
