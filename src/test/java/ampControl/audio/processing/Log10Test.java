package ampControl.audio.processing;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

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
        final ProcessingResult.Factory proc = new Log10();
        final ProcessingResult res = proc.create(new SingletonDoubleInput(input));
        assertEquals("Incorrect output size!", 1, res.get().size());
        assertEquals("Incorrect output size!", 1, res.get().get(0).length);
        assertArrayEquals("Incorrect ouput!", expected, res.get().get(0)[0], 1e-10);
    }
}