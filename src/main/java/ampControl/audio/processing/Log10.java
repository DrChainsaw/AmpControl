package ampControl.audio.processing;

import java.util.Collections;
import java.util.List;

/**
 * Does base 10 logarithm of input.
 *
 * @author Christian Skärby
 */
public class Log10 implements ProcessingResult.Processing {

    private double[][] result;
    private static final double minValid = 1e-10;

    @Override
    public void receive(double[][] input) {
        final int nrofFrames = input.length;
        final int nrofSamplesPerBin = input[0].length;
        this.result = new double[nrofFrames][nrofSamplesPerBin];

        for (int i = 0; i < nrofFrames; i++) {
            for (int j = 0; j < nrofSamplesPerBin; j++) {
                if (input[i][j] < minValid) { // Avoid NaN/-inf
                    this.result[i][j] = Math.log10(minValid);
                } else {
                    this.result[i][j] = Math.log10(input[i][j]);
                }
            }
        }
    }

    @Override
    public String name() {
        return nameStatic();
    }


    public static String nameStatic() {
        return "lg10";
    }

    @Override
    public List<double[][]> get() {
        return Collections.singletonList(result);
    }
}
