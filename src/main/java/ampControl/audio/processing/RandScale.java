package ampControl.audio.processing;

import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Does random scaling of input. Useful for data augmentation. Obviously useless if normalization is applied. Should also
 * obviously not be recreated when doing classification of unknown input.
 *
 * @author Christian Skärby
 */
public class RandScale implements ProcessingResult.Processing {

    private final static String name = "rdsc";
    private final static String upperPrefix = "_u";
    private final static String lowerPrefix = "_l";

    private final int maxScalingPerc;
    private final int minScalingPerc;
    private final Random rng;

    private double[][] result;

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
            throw new RuntimeException("Upper must not be smaller than lower!");
        }
    }

    @Override
    public void receive(double[][] input) {
        final int nrofFrames = input.length;
        final int nrofSamplesPerFrame = input[0].length;
        this.result= new double[nrofFrames][nrofSamplesPerFrame];
        final double scalingFactor = (minScalingPerc + rng.nextInt(maxScalingPerc - minScalingPerc)) / 1e2;

        for (int i = 0; i < nrofFrames; i++) {
            for (int j = 0; j < nrofSamplesPerFrame; j++) {
                this.result[i][j] = input[i][j]*scalingFactor;
            }
        }
    }

    @Override
    public String name() {
        return name + upperPrefix + maxScalingPerc + lowerPrefix + minScalingPerc;
    }

    @Override
    public List<double[][]> get() {
        return Collections.singletonList(result);
    }
}
