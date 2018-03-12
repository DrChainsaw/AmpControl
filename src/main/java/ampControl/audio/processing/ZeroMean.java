package ampControl.audio.processing;

import java.util.Collections;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * Removes mean from input.
 *
 * @author Christian Skärby
 */
public class ZeroMean implements ProcessingResult.Processing {

    private double[][] result;

    @Override
    public void receive(double[][] input) {
        final int nrofFrames = input.length;
        final int nrofSamplesPerFrame = input[0].length;

        this.result= new double[nrofFrames][nrofSamplesPerFrame];
        final double avg = Stream.of(input).flatMapToDouble(dVec -> DoubleStream.of(dVec)).summaryStatistics().getAverage();

        for (int i = 0; i < nrofFrames; i++) {
            for (int j = 0; j < nrofSamplesPerFrame; j++) {
                this.result[i][j] = input[i][j] - avg;
            }
        }
    }

    @Override
    public String name() {
        return nameStatic();
    }

    public static String nameStatic() {
        return "zm";
    }

    @Override
    public List<double[][]> get() {
        return Collections.singletonList(result);
    }
}
