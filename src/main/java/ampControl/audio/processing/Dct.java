package ampControl.audio.processing;

import ampControl.model.visualize.RealTimePlot;
import org.jtransforms.dct.DoubleDCT_1D;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Does discrete cosine transform of input
 *
 * @author Christian Skärby
 */
public class Dct implements ProcessingResult.Processing {

    private double[][] dct;

    @Override
    public void receive(double[][] input) {
        final int nrofFrames = input.length;
        final int nrofSamplesPerBin = input[0].length;
        this.dct= new double[nrofFrames][nrofSamplesPerBin];
        // Could be cached or initialized in some factory but it does not seem to be worth it.
        DoubleDCT_1D dct1d = new DoubleDCT_1D(nrofSamplesPerBin);
        for (int i = 0; i < dct.length; i++) {
            dct[i] = input[i].clone();
            dct1d.forward(dct[i], false);
        }
    }

    @Override
    public List<double[][]> get() {
        return Collections.singletonList(dct);
    }

    @Override
    public String name() {
        return nameStatic();
    }

    public static String nameStatic() {
        return "dct";
    }

    public static void main(String[] args) {
        final Dct dct = new Dct();
        final int size = 1024;
        List<Integer> freqs = Arrays.asList(17, 50);
        double[] cosSinSum = IntStream.range(0, size)
                .mapToDouble(i -> i * 2 * Math.PI / size)
                .map(d -> freqs.stream().mapToDouble(freq -> Math.cos(freq*d)).sum() + Math.sin(5*d))
                .toArray();
        dct.receive(new double[][] {cosSinSum});
        double[] dctData = dct.get().get(0)[0];
        RealTimePlot<Integer, Double> rtp = new RealTimePlot<>("dct", "dummy");
        for(int i = 0; i < cosSinSum.length; i++) {
           rtp.plotData("dct", i, dctData[i]/size);
            rtp.plotData("cos", i, cosSinSum[i]);
        }
    }
}
