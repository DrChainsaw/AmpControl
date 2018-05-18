package ampcontrol.audio.processing;

import java.util.DoubleSummaryStatistics;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * (Pseudo) normalizes the input to have unit abs max and zero mean
 *
 * @author Christian Skärby
 */
public class UnitMaxZeroMean implements ProcessingResult.Factory {

    @Override
    public String name() {
        return nameStatic();
    }


    public static String nameStatic() {
        return "umzm";
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
                final double[][] scaled = new double[nrofFrames][nrofSamplesPerFrame];

                final DoubleSummaryStatistics stats = Stream.of(inputArr).flatMapToDouble(DoubleStream::of).summaryStatistics();
                final double min = stats.getMin();
                final double max = stats.getMax();
                final double avg = stats.getAverage();

                final double absMax = Math.max(1e-10, Math.max(Math.abs(max - avg), Math.abs(min - avg)));
                for (int i = 0; i < nrofFrames; i++) {
                    for (int j = 0; j < nrofSamplesPerFrame; j++) {
                        scaled[i][j] = (inputArr[i][j] - avg) / absMax;
                    }
                }
                return scaled;
            });
        }
    }
}
