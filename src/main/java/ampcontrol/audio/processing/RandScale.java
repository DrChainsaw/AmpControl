package ampcontrol.audio.processing;

import java.util.Random;
import java.util.stream.Stream;

/**
 * Does random scaling of input. Useful for data augmentation. Obviously useless if normalization is applied. Should also
 * obviously not be recreated when doing classification of unknown input.
 *
 * @author Christian Skärby
 */
public class RandScale implements ProcessingResult.Factory {

    private final static String name = "rdsc";
    private final static String upperPrefix = "_u";
    private final static String lowerPrefix = "_l";

    private final int maxScalingPerc;
    private final int minScalingPerc;
    private final Random rng;


    /**
     * Constructor
     * @param maxScalingPerc Maximum (largest) possible scaling in percent (i.e 120 means 120% i.e. 20% more than input)
     * @param minScalingPerc Minimum (smallest) possible scaling in precent (i.e 80 means 80% i.e 20% less than input)
     * @param rng Random number generator.
     */
    public RandScale(int maxScalingPerc, int minScalingPerc, Random rng) {
        this.maxScalingPerc = maxScalingPerc;
        this.minScalingPerc = minScalingPerc;
        this.rng = rng;

        if(maxScalingPerc < minScalingPerc) {
            throw new IllegalArgumentException("Upper must not be smaller than lower!");
        }
    }

    @Override
    public String name() {
        return name + upperPrefix + maxScalingPerc + lowerPrefix + minScalingPerc;
    }

    @Override
    public ProcessingResult create(ProcessingResult input) {
        return new Result(input);
    }

    private final class Result implements ProcessingResult {

        private final ProcessingResult input;

        public Result(ProcessingResult input) {
            this.input = input;
        }

        @Override
        public Stream<double[][]> stream() {
            return input.stream().map(inputArr -> {
                final int nrofFrames = inputArr.length;
                final int nrofSamplesPerFrame = inputArr[0].length;
                final double[][] result = new double[nrofFrames][nrofSamplesPerFrame];
                final double scalingFactor = (minScalingPerc + rng.nextInt(maxScalingPerc - minScalingPerc)) / 1e2;

                for (int i = 0; i < nrofFrames; i++) {
                    for (int j = 0; j < nrofSamplesPerFrame; j++) {
                        result[i][j] = inputArr[i][j] * scalingFactor;
                    }
                }
                return result;
            });
        }
    }
}
