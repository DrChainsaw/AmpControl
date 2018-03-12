package ampControl.audio.processing;

import org.junit.Test;

import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import static org.junit.Assert.*;

/**
 * Test cases for {@link LogScale}
 *
 * @author Christian Skärby
 */
public class LogScaleTest {

    /**
     * Test that scaling works as expected
     */
    @Test
    public void receive() {
        final int maxPow = 10;
        final double[] expected = IntStream.rangeClosed(0,maxPow).mapToDouble(i -> i/maxPow).toArray();
        final double[] tenPows = DoubleStream.of(expected).map(pow -> Math.pow(10, pow*maxPow)).toArray();
        final ProcessingResult.Processing logScale = new LogScale();
        logScale.receive(new double[][] {tenPows});
        assertArrayEquals("Incorrect output!", expected, logScale.get().get(0)[0], 1e-10);
    }

    /**
     * Test that scaling works as expected
     */
    @Test
    public void receiveDescending() {
        final int maxPow = 10;
        final double[] expected = IntStream.rangeClosed(0,maxPow).map(i -> maxPow-i).mapToDouble(i -> i/maxPow).toArray();
        final double[] tenPows = DoubleStream.of(expected).map(pow -> Math.pow(10, pow*maxPow)).toArray();
        final ProcessingResult.Processing logScale = new LogScale();
        logScale.receive(new double[][] {tenPows});
        assertArrayEquals("Incorrect output!", expected, logScale.get().get(0)[0], 1e-10);
    }

    /**
     * Test that naming is consistent
     */
    @Test
    public void name() {
        assertEquals("Incorrect name!", LogScale.nameStatic(), new LogScale().name());
    }
}