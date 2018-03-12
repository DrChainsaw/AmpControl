package ampControl.audio.processing;

import java.util.Collections;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * Does logarithmic scaling of input y = log10(x / minX) / log10(maxX / minX). Extracted from
 * {@link org.datavec.audio.extension.Spectrogram}.
 *
 * @author Jacquet Wong
 */
public class LogScale implements ProcessingResult.Processing {

    private double[][] scaled;

    @Override
    public void receive(double[][] input) {
        final int nrofFrames = input.length;
        final int nrofSamplesPerBin = input[0].length;
        this.scaled = new double[nrofFrames][nrofSamplesPerBin];


        final DoubleSummaryStatistics stats = Stream.of(input).flatMapToDouble(dVec -> DoubleStream.of(dVec)).summaryStatistics();
        final double min = stats.getMin() != 0 ? stats.getMin() :  0.00000000001F;
        final double max = stats.getMax();

        final double diff = Math.log10(max / min); // perceptual difference
        for (int i = 0; i < nrofFrames; i++) {
            for (int j = 0; j < nrofSamplesPerBin; j++) {
                if (input[i][j] < min) {
                    scaled[i][j] = 0;
                } else {
                    scaled[i][j] = Math.log10(input[i][j] / min) / diff;
                }
            }
        }
    }

    @Override
    public List<double[][]> get() {
        return Collections.singletonList(scaled);
    }

    @Override
    public String name() {
        return nameStatic();
    }

    public static String nameStatic() {
        return "lgsc";
    }
}
