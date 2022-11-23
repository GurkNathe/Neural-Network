package Functions;

import java.util.Random;

public class InitialWeights {
    public enum InitialWeightsType {
        Xavier
    }

    public static InitialWeightsFunction getInitialWeightsFunction(InitialWeightsType initialWeightsType) {
        switch (initialWeightsType) {
            case Xavier:
                return new Xavier();
            default:
                return new Xavier();
        }
    }

    static class Xavier implements InitialWeightsFunction {
        /**
         * @param values :
         *               [0] -> the number of incoming network connections
         *               [1] -> the number of outgoing network connections from that
         *               layer
         * @return double value in the range
         *         [ -√6/(value[0] + value[1]), √6/(value[0] + value[1]) )
         */
        public double getInitialWeight(double... values) {
            // Upperbound for distribution
            double upperBound = Math.sqrt(6) / (Math.sqrt(values[0] + values[1]));

            Random rng = new Random();

            // Get integer to decide sign of value
            int sign = rng.nextInt(2);

            // Value to be scaled by the upper bound [0, 1)
            double value = rng.nextDouble();

            return sign == 1 ? upperBound * value : -upperBound * value;
        }
    }
}
