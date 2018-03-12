package ampControl.audio.processing;

import java.util.Collections;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * Normalizes the input to unit max and zero min.
 *
 * @author Christian Skärby
 */
public class UnitMaxZeroMin implements ProcessingResult.Processing {


    private double[][] scaled;

    @Override
    public void receive(
            double[][] input) {
        final int nrofFrames = input.length;
        final int nrofSamplesPerFrame = input[0].length;
        this.scaled = new double[nrofFrames][nrofSamplesPerFrame];

        final DoubleSummaryStatistics stats = Stream.of(input).flatMapToDouble(dVec -> DoubleStream.of(dVec)).summaryStatistics();
        final double min = stats.getMin();
        final double max = stats.getMax();

        double diff = Math.max(1e-10, max - min);
        for (int i = 0; i < nrofFrames; i++) {
            for (int j = 0; j < nrofSamplesPerFrame; j++) {
                scaled[i][j] = (input[i][j] - min) / diff;
            }
        }
    }

    @Override
    public String name() {
        return nameStatic();
    }

    public static String nameStatic() {
        return "norm";
    }

    @Override
    public List<double[][]> get() {
        return Collections.singletonList(scaled);
    }
}
