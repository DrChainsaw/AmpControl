package ampControl.audio.processing;

import java.util.Collections;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * (Pseudo) normalizes the input to have unit abs max and zero mean
 *
 * @author Christian Skärby
 */
public class UnitMaxZeroMean implements ProcessingResult.Processing {

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
        final double avg = stats.getAverage();

        final double absMax = Math.max(1e-10, Math.max(Math.abs(max - avg), Math.abs(min - avg)));
        for (int i = 0; i < nrofFrames; i++) {
            for (int j = 0; j < nrofSamplesPerFrame; j++) {
                scaled[i][j] = (input[i][j] - avg) / absMax;
            }
        }
    }

    @Override
    public String name() {
        return nameStatic();
    }

    @Override
    public List<double[][]> get() {
        return Collections.singletonList(scaled);
    }

    public static String nameStatic() {
        return "umzm";
    }
}
