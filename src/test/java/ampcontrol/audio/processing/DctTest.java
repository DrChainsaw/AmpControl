package ampcontrol.audio.processing;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Test cases for {@link Dct}
 *
 * @author Christian Skärby
 */
public class DctTest {

    /**
     * Test that output resembles a DCT
     */
    @Test
    public void receive(){
        final int size = 1024;
        List<Integer> freqs = Arrays.asList(17, 23);
        double[] cosSinSum = IntStream.range(0, size)
                .mapToDouble(i -> i * 2 * Math.PI / size)
                .map(d -> freqs.stream().mapToDouble(freq -> Math.cos(freq*d)).sum() + Math.sin(5*d)) // sin is supressed
                .toArray();

        final Dct dct = new Dct();
        final ProcessingResult res = dct.create(new SingletonDoubleInput(cosSinSum));

        final double[] dctData = res.stream().findFirst().get()[0];

        for(int i = 0; i < freqs.size(); i++) {
            final int argmax = argMax(dctData);
            dctData[argmax] = 0;
            final Integer actualFreq = argmax / 2;
            assertTrue("Frequency component " + actualFreq + " in DCT not present in input signal!", freqs.contains(actualFreq));
        }
    }


    /**
     * Test that naming is consistent
     */
    @Test
    public void name() {
        assertEquals("Name must be equal to static name!", Dct.nameStatic(), new Dct().name());
    }

    private static int argMax(double[] in) {
        int maxIdx = 0;

        for(int i = 1; i < in.length; ++i) {
            if (in[i] > in[maxIdx]) {
                maxIdx = i;
            }
        }

        return maxIdx;
    }
}