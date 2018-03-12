package ampControl.audio.processing;

import java.util.Collections;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * Standardizes the input to have unit standard deviation and zero mean.
 *
 * @author Christian Skärby
 */
public class UnitStdZeroMean implements ProcessingResult.Processing {

    private double[][] scaled;

    @Override
    public void receive(double[][] input) {
        final int nrofFrames = input.length;
        final int nrofSamplesPerFrame = input[0].length;
        final int totalNrofSamples = nrofFrames * nrofSamplesPerFrame;
        this.scaled = new double[nrofFrames][nrofSamplesPerFrame];
        final double avg = Stream.of(input).flatMapToDouble(dVec -> DoubleStream.of(dVec)).summaryStatistics().getAverage();

        double varSum = 0;
        for (int i = 0; i < nrofFrames; i++) {
            for (int j = 0; j < nrofSamplesPerFrame; j++) {
                this.scaled[i][j] = input[i][j] - avg;
                varSum += this.scaled[i][j] * this.scaled[i][j];
            }
        }
        final double std = Math.max(1e-10, Math.sqrt(varSum / totalNrofSamples));
        for (int i = 0; i < nrofFrames; i++) {
            for (int j = 0; j < nrofSamplesPerFrame; j++) {
                this.scaled[i][j] /= std;
            }
        }
    }


    @Override
    public List<double[][]> get() {
        return Collections.singletonList(scaled);
    }

    @Override
    public String name() {
        return nameStatic();
    }

    public static String nameStatic() {
        return "uszm";
    }

}
