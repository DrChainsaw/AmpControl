package ampcontrol.audio.processing;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * {@link ProcessingResult.Factory} which buffers input.
 *
 * @author Christian Skärby
 */
public class Buffer implements ProcessingResult.Factory {

    @Override
    public String name() {
        return nameStatic();
    }

    public static String nameStatic() {
        return "buff";
    }

    @Override
    public ProcessingResult create(ProcessingResult input) {
        return new Result(input);
    }

    private final static class Result implements ProcessingResult {

        private final ProcessingResult input;
        private List<double[][]> bufferedInput;

        private Result(ProcessingResult input) {
            this.input = input;
        }

        @Override
        public Stream<double[][]> stream() {
            if(bufferedInput == null) {
                bufferedInput = input.stream()
                        .map(Result::deepCopy)
                        .collect(Collectors.toList());
            }
            return bufferedInput.stream()
                    .map(Result::deepCopy);

        }

        private static double[][] deepCopy(double[][] array) {
            double[][] copy = new double[array.length][array[0].length];
            for(int i = 0; i < array.length; i++) {
                System.arraycopy(array[i], 0, copy[i], 0, array[i].length);
            }
            return copy;
        }
    }
}
