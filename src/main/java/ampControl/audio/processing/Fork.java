package ampControl.audio.processing;

import java.util.ArrayList;
import java.util.List;

/**
 * Forks the processing path, resulting in two paths. The result of each unique path
 * will be one member of the result list. Intended use is to have multiple versions
 * of the input with different processing as input to the classifier, e.g. as
 * different channels.
 *
 * @author Christian Skärby
 */
public class Fork implements ProcessingResult.Processing {

    private final ProcessingResult.Processing path1;
    private final Processing path2;

    public Fork(Processing path1, Processing path2) {
        this.path1 = path1;
        this.path2 = path2;
    }

    @Override
    public void receive(double[][] input) {
        path1.receive(input);
        path2.receive(input);
    }


    @Override
    public List<double[][]> get() {
        List<double[][]> result = new ArrayList<>();
        result.addAll(path1.get());
        result.addAll(path2.get());
        return result;
    }

    @Override
    public String name() {
        return fork + path1.name() + split + path2.name() + krof;
    }

    public static String[] splitFirst(String str) {
        return str.split(fork, 2);
    }

    public static String[] splitMid(String str) {
        int forkInd = str.indexOf(fork);
        int splitInd = str.indexOf(split);
        if (forkInd != -1 && forkInd < splitInd) {
            String[] div = splitFirst(str);
            String[] divSplit = splitMid(div[1]);
            String[] divSplitSplit = splitMid(divSplit[1]);
            return new String[]{div[0] + fork + divSplit[0] + split +  divSplitSplit[0], divSplitSplit[1]};
        }
        return str.split(split, 2);
    }

    public static String[] splitEnd(String str) {
        return splitLastDelim(str, krof);
    }

    private static String[] splitLastDelim(String str, String delim) {
        final int splitInd = str.lastIndexOf(delim);
        final String before = str.substring(0, splitInd);
        final String after = str.substring(splitInd + delim.length());
        return new String[]{before, after};
    }


    public static String matchStrStatic() {
        return fork + ".*" + split + ".*" + krof;
    }

    final static String fork = "fork_";
    final static String split = "_split_";
    final static String krof = "_krof";
}
