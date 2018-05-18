package ampcontrol.audio.processing;

import java.util.DoubleSummaryStatistics;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * Does logarithmic scaling of input y = log10(x / minX) / log10(maxX / minX). Extracted from
 * {@link org.datavec.audio.extension.Spectrogram}.
 *
 * @author Jacquet Wong
 */
public class LogScale implements ProcessingResult.Factory {

    @Override
    public String name() {
        return nameStatic();
    }

    public static String nameStatic() {
        return "lgsc";
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
                final double[][] scaled = new double[nrofFrames][nrofSamplesPerBin];


                final DoubleSummaryStatistics stats = Stream.of(inputArr).flatMapToDouble(DoubleStream::of).summaryStatistics();
                final double min = stats.getMin() != 0 ? stats.getMin() : 0.00000000001F;
                final double max = stats.getMax();

                final double diff = Math.log10(max / min); // perceptual difference
                for (int i = 0; i < nrofFrames; i++) {
                    for (int j = 0; j < nrofSamplesPerBin; j++) {
                        if (inputArr[i][j] < min) {
                            scaled[i][j] = 0;
                        } else {
                            scaled[i][j] = Math.log10(inputArr[i][j] / min) / diff;
                        }
                    }
                }
                return scaled;
            });
        }
    }
}
