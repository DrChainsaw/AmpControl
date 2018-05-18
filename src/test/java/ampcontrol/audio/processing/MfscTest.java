package ampcontrol.audio.processing;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link Mfsc}
 *
 * @author Christian Skärby
 */
public class MfscTest {

    /**
     * Test that the output is same as "expected"
     */
    @Test
    public void receive() {
        // Don't know how what to expect really...
        final double[][] test = {{10,1,1,10,1,1,10}, {1,1,10,1,10,1,1}};
        final double[][] expected = {{2.303, 2.398, 0.0, 0.0, 0.693, 2.398, 2.303}, {0.0, 0.693, 0.0, 0.0, 2.398, 2.398, 0.0}};

        final ProcessingResult.Factory mfsc = new Mfsc(10000);
        final ProcessingResult res = mfsc.create(new SingletonDoubleInput(test));
        double[][] result = res.stream().findFirst().get();
        assertEquals("Incorrect size!", expected.length, result.length);
        for(int i = 0; i < expected.length; i++) {
            assertArrayEquals("Incorrect output!", expected[i], result[i], 1e-3);
        }
    }

    @Test
    public void name() {
        assertEquals("Inconsistent name!", Mfsc.nameStatic(), new Mfsc(100).name());
    }
}