package ampControl.audio.processing;

import ampControl.model.visualize.RealTimePlot;
import org.jtransforms.dct.DoubleDCT_1D;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Does discrete cosine transform of input
 *
 * @author Christian Skärby
 */
public class Dct implements ProcessingResult.Factory {

    @Override
    public String name() {
        return nameStatic();
    }

    public static String nameStatic() {
        return "dct";
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
                final double[][] dct = new double[nrofFrames][nrofSamplesPerBin];
                // Could be cached or initialized in the factory but it does not seem to be worth it.
                final DoubleDCT_1D dct1d = new DoubleDCT_1D(nrofSamplesPerBin);
                for (int i = 0; i < dct.length; i++) {
                    dct[i] = inputArr[i].clone();
                    dct1d.forward(dct[i], false);
                }
                return dct;
            });
        }
    }

    public static void main(String[] args) {
        final Dct dct = new Dct();
        final int size = 1024;
        List<Integer> freqs = Arrays.asList(17, 50);
        double[] cosSinSum = IntStream.range(0, size)
                .mapToDouble(i -> i * 2 * Math.PI / size)
                .map(d -> freqs.stream().mapToDouble(freq -> Math.cos(freq*d)).sum() + Math.sin(5*d))
                .toArray();
        ProcessingResult res = dct.create(new SingletonDoubleInput(cosSinSum));
        double[] dctData = res.stream().findAny().get()[0];
        RealTimePlot<Integer, Double> rtp = new RealTimePlot<>("dct", "dummy");
        for(int i = 0; i < cosSinSum.length; i++) {
           rtp.plotData("dct", i, dctData[i]/size);
            rtp.plotData("cos", i, cosSinSum[i]);
        }
    }
}
