package ampcontrol.audio.processing;

import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * Standardizes the input to have unit standard deviation and zero mean.
 *
 * @author Christian Skärby
 */
public class UnitStdZeroMean implements ProcessingResult.Factory {

    @Override
    public String name() {
        return nameStatic();
    }

    public static String nameStatic() {
        return "uszm";
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
                final int totalNrofSamples = nrofFrames * nrofSamplesPerFrame;
                final double[][] scaled = new double[nrofFrames][nrofSamplesPerFrame];
                final double avg = Stream.of(inputArr).flatMapToDouble(DoubleStream::of).summaryStatistics().getAverage();

                double varSum = 0;
                for (int i = 0; i < nrofFrames; i++) {
                    for (int j = 0; j < nrofSamplesPerFrame; j++) {
                        scaled[i][j] = inputArr[i][j] - avg;
                        varSum += scaled[i][j] * scaled[i][j];
                    }
                }
                final double std = Math.max(1e-10, Math.sqrt(varSum / totalNrofSamples));
                for (int i = 0; i < nrofFrames; i++) {
                    for (int j = 0; j < nrofSamplesPerFrame; j++) {
                        scaled[i][j] /= std;
                    }
                }
                return scaled;
            });
        }
    }
}
