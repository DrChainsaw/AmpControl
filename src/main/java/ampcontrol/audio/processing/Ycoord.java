package ampcontrol.audio.processing;

import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Outputs coordinates (indexes) for dimension 1 of input. Idea is that this helps mitigating undesired invariance.
 * Dim 1 from {@link Spectrogram} output is frequency and thus frequency invariance is reduced. Not yet tested if this
 * is beneficial or not though.
 * <br><br>
 * https://arxiv.org/abs/1807.07044
 * https://arxiv.org/abs/1807.03247
 *
 * @author Christian Skärby
 */
public class Ycoord implements ProcessingResult.Factory {

    @Override
    public String name() {
        return "ycoord";
    }

    public static String nameStatic() {
        return new Ycoord().name();
    }

    @Override
    public ProcessingResult create(ProcessingResult input) {
        return new Result(input);
    }

    private class Result implements ProcessingResult {
        private final ProcessingResult input;

        private Result(ProcessingResult input) {
            this.input = input;
        }

        @Override
        public Stream<double[][]> stream() {
            return input.stream().map(inputArr -> {
                final int yLen = inputArr[0].length;
                final double[] proto = IntStream.range(0, inputArr[0].length).mapToDouble(i -> i).toArray();
                final double[][] result = new double[inputArr.length][yLen];
                for(int i = 0; i < inputArr.length; i++) {
                    result[i] = proto;
                }
                return result;
            });
        }
    }
}
