package ampcontrol.model.training.data.state;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * Test cases for {@link ResetableRandomState}
 */
public class RandomStateControllerTest {

    /**
     * Test that state can be stored and reset
     */
    @Test
    public void testReset() {
        final long seed = 666;
        final int sequenceLength = 7;
        final ResetableRandomState controller = new ResetableRandomState(seed);

        final int[] expected = controller.getRandom().ints().limit(sequenceLength).toArray();
        controller.restorePreviousState();
        final int[] actual = controller.getRandom().ints().limit(sequenceLength).toArray();
        assertArrayEquals("Incorrect sequence!", expected, actual);

        controller.storeCurrentState();
        final int[] newExpected = controller.getRandom().ints().limit(sequenceLength).toArray();
        assertNotEquals("Expected different sequence after new state!", Arrays.toString(expected), Arrays.toString(newExpected));
        controller.restorePreviousState();
        final int[] newActual = controller.getRandom().ints().limit(sequenceLength).toArray();
        assertArrayEquals("Incorrect sequence!", newExpected, newActual);

    }
}