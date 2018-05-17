package ampcontrol.audio.processing;

import org.junit.Test;

import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

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
        final ProcessingResult.Factory logScale = new LogScale();
        final ProcessingResult res = logScale.create(new SingletonDoubleInput(tenPows));
        assertArrayEquals("Incorrect output!", expected, res.stream().findFirst().get()[0], 1e-10);
    }

    /**
     * Test that scaling works as expected
     */
    @Test
    public void receiveDescending() {
        final int maxPow = 10;
        final double[] expected = IntStream.rangeClosed(0,maxPow).map(i -> maxPow-i).mapToDouble(i -> i/maxPow).toArray();
        final double[] tenPows = DoubleStream.of(expected).map(pow -> Math.pow(10, pow*maxPow)).toArray();
        final ProcessingResult.Factory logScale = new LogScale();
        final ProcessingResult res = logScale.create(new SingletonDoubleInput(tenPows));
        assertArrayEquals("Incorrect output!", expected, res.stream().findFirst().get()[0], 1e-10);
    }

    /**
     * Test that naming is consistent
     */
    @Test
    public void name() {
        assertEquals("Incorrect name!", LogScale.nameStatic(), new LogScale().name());
    }
}