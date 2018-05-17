package ampcontrol.audio.processing;

import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * Removes mean from input.
 *
 * @author Christian Skärby
 */
public class ZeroMean implements ProcessingResult.Factory {

    @Override
    public String name() {
        return nameStatic();
    }

    public static String nameStatic() {
        return "zm";
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
                final double avg = Stream.of(inputArr).flatMapToDouble(DoubleStream::of).summaryStatistics().getAverage();

                for (int i = 0; i < nrofFrames; i++) {
                    for (int j = 0; j < nrofSamplesPerFrame; j++) {
                        result[i][j] = inputArr[i][j] - avg;
                    }
                }
                return result;
            });
        }
    }
}
