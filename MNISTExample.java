import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

import java.io.InputStream;
import java.net.URL;
import java.util.zip.GZIPInputStream;

import MNIST.MnistImageFile;
import MNIST.MnistLabelFile;
import MNIST.MnistPanel;

public class MNISTExample {
    public static void main(String[] args) throws FileNotFoundException, IOException, InterruptedException {

        // ------- Load MNIST data -------
        String path = new File("").getAbsolutePath();

        MnistImageFile image = new MnistImageFile(path + "/MNIST/train-images.idx3-ubyte", "rw");
        MnistLabelFile label = new MnistLabelFile(path + "/MNIST/train-labels.idx1-ubyte", "rw");

        // ------- ------- ------- -------

        // ------- Initialize Hyper Parameters -------
        int[] layers = { 784, 200, 10 };

        double[][] initialWeights = new double[layers.length - 1][2];

        for (int i = 0; i < initialWeights.length; i++) {
            for (int j = 0; j < initialWeights[i].length; j++) {
                initialWeights[i][j] = layers[i + j];
            }
        }

        HyperParameter data = new HyperParameter(
                Activation.ActivationFunctionType.Sigmoid,
                InitialWeights.InitialWeightsType.Xavier,
                Cost.CostFunctionType.MeanSquaredError,
                initialWeights,
                layers,
                0.05,
                0.075,
                0.9,
                0.1);

        // ------- ------- ------- ------- ------- -------

        // Initialize network
        Network network = new Network(data);

        // Load the MNIST data into memory
        double[][] image_buffers = new double[60000][784];
        for (int i = 0; i < 60000; i++) {
            for (int j = 0; j < 784; j++) {
                image_buffers[i][j] = image.read();
            }
        }

        // Create expected outputs
        double[][] expectedOutputs = new double[60000][10];
        for (int i = 0; i < 60000; i++) {
            int expected = label.readLabel();
            expectedOutputs[i][expected] = 1;
        }

        Data networkData = new Data(image_buffers, expectedOutputs);
        // DataPoint[] networkData = new DataPoint[networkData.

        // network.learn(networkData, 0.001);

        image.close();
        label.close();
    }
}
