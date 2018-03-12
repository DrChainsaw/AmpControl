package ampControl.audio.processing;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Masks out indexes aling the second dimension, setting them to 0.
 *
 * @author Christian Skärby
 */
public class MaskBins implements ProcessingResult.Processing {

    private final int[] binsToMask;
    private double[][] result;

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

    @Override
    public void receive(double[][] input) {
        result = input.clone();
        for(int i = 0; i < result.length; i++) {
            for(int j: binsToMask) {
                result[i][j] = 0;
            }
        }
    }

    @Override
    public String name() {
        return nameStatic() + Arrays.stream(binsToMask)
                .mapToObj(i -> String.valueOf(i))
                .collect(Collectors.joining(delim()));
    }

    public static String nameStatic() {
        return "mask";
    }

    private static String delim() {
        return "c";
    }

    @Override
    public List<double[][]> get() {
        return Collections.singletonList(result);
    }

    public static void main(String[] args) {
        System.out.println("Created: " + new MaskBins("mask0c1c2c3c4").name());

    }
}
