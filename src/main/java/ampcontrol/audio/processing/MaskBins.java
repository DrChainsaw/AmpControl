package ampcontrol.audio.processing;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Masks out indexes along the second dimension, setting them to 0.
 *
 * @author Christian Skärby
 */
public class MaskBins implements ProcessingResult.Factory {

    private final int[] binsToMask;

    public MaskBins(int[] binsToMask) {
        this.binsToMask = binsToMask;
    }

    public MaskBins(String parStr) {
        String[] chop = parStr.split(nameStatic());
        if(chop.length != 2) {
            throw new IllegalArgumentException("Can't create mask from! " + parStr);
        }
        binsToMask = Arrays.stream(chop[1].split(delim()))
                .mapToInt(Integer::parseInt)
                .toArray();

    }

    public static String nameStatic() {
        return "mask";
    }

    private static String delim() {
        return "c";
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
                double[][] result = inputArr.clone();
                for (int i = 0; i < result.length; i++) {
                    for (int j : binsToMask) {
                        result[i][j] = 0;
                    }
                }
                return result;
            });
        }
    }

    @Override
    public String name() {
        return nameStatic() + Arrays.stream(binsToMask)
                .mapToObj(String::valueOf)
                .collect(Collectors.joining(delim()));
    }


    public static void main(String[] args) {
        System.out.println("Created: " + new MaskBins("mask0c1c2c3c4").name());
    }
}
