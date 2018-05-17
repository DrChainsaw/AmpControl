package ampcontrol.audio.processing;

import java.util.stream.Stream;

/**
 * Does base 10 logarithm of input.
 *
 * @author Christian Skärby
 */
public class Log10 implements ProcessingResult.Factory {

    private static final double minValid = 1e-10;


    @Override
    public String name() {
        return nameStatic();
    }

    public static String nameStatic() {
        return "lg10";
    }

    @Override
    public ProcessingResult create(ProcessingResult input) {
        return new Result(input);
    }

    private final static class Result implements ProcessingResult {

        private final ProcessingResult input;

        public Result(ProcessingResult input) {
            this.input = input;
        }

        @Override
        public Stream<double[][]> stream() {
            return input.stream().map(inputArr -> {
            final int nrofFrames = inputArr.length;
            final int nrofSamplesPerBin = inputArr[0].length;
            double[][] result = new double[nrofFrames][nrofSamplesPerBin];

            for (int i = 0; i < nrofFrames; i++) {
                for (int j = 0; j < nrofSamplesPerBin; j++) {
                    if (inputArr[i][j] < minValid) { // Avoid NaN/-inf
                        result[i][j] = Math.log10(minValid);
                    } else {
                        result[i][j] = Math.log10(inputArr[i][j]);
                    }
                }
            }
            return result;
            });
        }
    }
}
