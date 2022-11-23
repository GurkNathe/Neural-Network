package Network;

public class Data {
    public double[][] trainedData;
    public double[][] expectedData;

    public Data(double[][] trainedData, double[][] expectedData) {
        this.trainedData = trainedData;
        this.expectedData = expectedData;
    }
}
