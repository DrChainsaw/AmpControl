package ampControl.audio.processing;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Test cases for {@link Log10}
 *
 * @author Christian Skärby
 */
public class Log10Test {

    /**
     * Test that naming is consistent
     */
    @Test
    public void name() {
        assertEquals("Incorrect name!", Log10.nameStatic(), new Log10().name());
    }

    @Test
    public void testProcessing() {
        final double[] input =   {  -1,   0, 1e0, 1e1, 1e2, 1e3, 1e4};
        final double[] expected = {-10, -10,   0,   1,   2,   3,   4};
        final ProcessingResult.Processing proc = new Log10();
        proc.receive(new double[][] {input});
        assertEquals("Incorrect output size!", 1, proc.get().size());
        assertEquals("Incorrect output size!", 1, proc.get().get(0).length);
        assertArrayEquals("Incorrect ouput!", expected, proc.get().get(0)[0], 1e-10);
    }
}