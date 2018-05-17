package ampcontrol.audio.processing;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link NoProcessing}
 *
 * @author Christian Skärby
 */
public class NoProcessingTest {

    /**
     * Test that output == input
     */
    @Test
    public void receive() {
        final double[][] expected = new double[][]{{1, 2, 3}, {4, 5, 6}};
        final ProcessingResult.Factory nopp = new NoProcessing();
        final ProcessingResult res = nopp.create(new SingletonDoubleInput(expected));
        assertArrayEquals("Incorrect output!", expected, res.stream().findFirst().get());
    }

    /**
     * Test that name is consistent
     */
    @Test
    public void name() {
        assertEquals("Inconsistent name!", NoProcessing.nameStatic(), new NoProcessing().name());
    }
}