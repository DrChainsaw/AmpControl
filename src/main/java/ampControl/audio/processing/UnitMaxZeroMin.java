package ampControl.audio.processing;

import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * Normalizes the input to unit max and zero min.
 *
 * @author Christian Skärby
 */
public class UnitMaxZeroMin implements ProcessingResult.Factory {


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
        public List<double[][]> get() {
            return input.get().stream().map(inputArr -> {
                final int nrofFrames = inputArr.length;
                final int nrofSamplesPerFrame = inputArr[0].length;
                final double[][] scaled = new double[nrofFrames][nrofSamplesPerFrame];

                final DoubleSummaryStatistics stats = Stream.of(inputArr).flatMapToDouble(dVec -> DoubleStream.of(dVec)).summaryStatistics();
                final double min = stats.getMin();
                final double max = stats.getMax();

                double diff = Math.max(1e-10, max - min);
                for (int i = 0; i < nrofFrames; i++) {
                    for (int j = 0; j < nrofSamplesPerFrame; j++) {
                        scaled[i][j] = (inputArr[i][j] - min) / diff;
                    }
                }
                return scaled;
            }).collect(Collectors.toList());
        }
    }

    @Override
    public String name() {
        return nameStatic();
    }

    public static String nameStatic() {
        return "norm";
    }
}
