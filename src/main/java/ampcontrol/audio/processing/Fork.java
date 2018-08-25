package ampcontrol.audio.processing;

import ampcontrol.audio.processing.ProcessingResult.Factory;

import java.util.stream.Stream;

/**
 * Forks the processing path, resulting in two paths. The result of each unique path
 * will be one member of the result list. Intended use is to have multiple versions
 * of the input with different processing as input to the classifier, e.g. as
 * different channels.
 *
 * @author Christian Skärby
 */
public class Fork implements ProcessingResult.Factory {

    private final static String fork = "fork_";
    private final static String split = "_split_";
    private final static String krof = "_krof";

    private final Factory path1;
    private final Factory path2;
    private final Buffer buffer;

    public Fork(Factory path1, ProcessingResult.Factory path2) {
        this.path1 = path1;
        this.path2 = path2;
        buffer = new Buffer();
    }

    @Override
    public ProcessingResult create(ProcessingResult input) {
        final ProcessingResult bufferedInput = buffer.create(input);
        return new Result(path1.create(bufferedInput), path2.create(bufferedInput));
    }

    private final static class Result implements ProcessingResult {

        private final ProcessingResult result1;
        private final ProcessingResult result2;

        public Result(ProcessingResult result1, ProcessingResult result2) {
            this.result1 = result1;
            this.result2 = result2;
        }

        @Override
        public Stream<double[][]> stream() {
            return Stream.concat(result1.stream(), result2.stream());
        }
    }

    @Override
    public String name() {
        return fork + path1.name() + split + path2.name() + krof;
    }

    static String[] splitFirst(String str) {
        return str.split(fork, 2);
    }

    static String[] splitMid(String str) {
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

    static String[] splitEnd(String str) {
        return splitLastDelim(str, krof);
    }

    private static String[] splitLastDelim(String str, String delim) {
        final int splitInd = str.lastIndexOf(delim);
        final String before = str.substring(0, splitInd);
        final String after = str.substring(splitInd + delim.length());
        return new String[]{before, after};
    }


    static String matchStrStatic() {
        return fork + ".*" + split + ".*" + krof;
    }

}
