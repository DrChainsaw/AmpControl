package ampControl.audio.processing;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

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
            throw new RuntimeException("Can't create mask from! " + parStr);
        }
        binsToMask = Arrays.stream(chop[1].split(delim()))
                .mapToInt(str -> Integer.parseInt(str))
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
        public List<double[][]> get() {
            return input.get().stream().map(inputArr -> {
                double[][] result = inputArr.clone();
                for (int i = 0; i < result.length; i++) {
                    for (int j : binsToMask) {
                        result[i][j] = 0;
                    }
                }
                return result;
            }).collect(Collectors.toList());
        }
    }

    @Override
    public String name() {
        return nameStatic() + Arrays.stream(binsToMask)
                .mapToObj(i -> String.valueOf(i))
                .collect(Collectors.joining(delim()));
    }


    public static void main(String[] args) {
        System.out.println("Created: " + new MaskBins("mask0c1c2c3c4").name());

    }
}
