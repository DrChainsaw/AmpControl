package ampControl.audio.processing;

import org.junit.Test;

import static org.junit.Assert.*;

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
        final ProcessingResult.Processing nopp = new NoProcessing();
        nopp.receive(expected);
        assertArrayEquals("Incorrect output!", expected, nopp.get().get(0));
    }

    /**
     * Test that name is consistent
     */
    @Test
    public void name() {
        assertEquals("Inconsistent name!", NoProcessing.nameStatic(), new NoProcessing().name());
    }
}