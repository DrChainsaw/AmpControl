package ampcontrol.audio.processing;

import org.junit.Test;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import static org.junit.Assert.*;

/**
 * Test cases for {@link RandScale}
 *
 * @author Christian Skärby
 */
public class RandScaleTest {

    /**
     * Test max scaling
     */
    @Test
    public void receiveUpperBound() {
        final int maxScaling = 133;
        final int minScaling = 77;
        final double [] input = {-2, 0, 3};
        final double [] expected = DoubleStream.of(input).map(d -> d * maxScaling / 1e2).toArray();
        final ProcessingResult.Factory proc = new RandScale(maxScaling, minScaling, new Random() {
            @Override
            public int nextInt(int bound) {
                return bound;
            }
        });
        final ProcessingResult res = proc.create(new SingletonDoubleInput(new double[][] {input}));
        final List<double[][]> resList = res.stream().collect(Collectors.toList());
        assertEquals("Incorrect output size!", 1, resList.size());
        assertEquals("Incorrect output size!", 1, resList.get(0).length);
        assertArrayEquals("Incorrect ouput!", expected, resList.get(0)[0], 1e-10);
    }

    /**
     * Test min scaling
     */
    @Test
    public void receiveLowerBound() {
        final int maxScaling = 777;
        final int minScaling = 13;
        final double [] input = {-2, 0, 3};
        final double [] expected = DoubleStream.of(input).map(d -> d * minScaling / 1e2).toArray();
        final ProcessingResult.Factory proc = new RandScale(maxScaling, minScaling, new Random() {
            @Override
            public int nextInt(int bound) {
                return 0;
            }
        });
        final ProcessingResult res = proc.create(new SingletonDoubleInput(new double[][] {input}));
        final List<double[][]> resList = res.stream().collect(Collectors.toList());
        assertEquals("Incorrect output size!", 1, resList.size());
        assertEquals("Incorrect output size!", 1, resList.get(0).length);
        assertArrayEquals("Incorrect ouput!", expected, resList.get(0)[0], 1e-10);
    }

    /**
     * Test that name is correct
     */
    @Test
    public void name() {
        final int maxP = 777;
        final int minP = 666;
        final String actual = new RandScale(maxP, minP, new Random()).name();
        assertTrue("Max scaling not part of name!", actual.matches(".*" + maxP + ".*"));
        assertTrue("Min scaling not part of name!", actual.matches(".*" + minP + ".*"));
    }

    /**
     * Test that mixup of max and min generates an exception
     */
    @Test(expected = RuntimeException.class)
    public void argMismatch() {
        new RandScale(10, 11, new Random());
    }
}