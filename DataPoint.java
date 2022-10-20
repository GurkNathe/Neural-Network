public class DataPoint {
    protected double[] inputs;
    protected double[] expectedOutputs;
    protected int label;

    public DataPoint(double[] inputs, int label, int numLabels) {
        this.inputs = inputs;
        this.label = label;
        expectedOutputs = CreateOneHot(label, numLabels);
    }

    public static double[] CreateOneHot(int index, int num) {
        double[] oneHot = new double[num];
        oneHot[index] = 1;
        return oneHot;
    }
}
