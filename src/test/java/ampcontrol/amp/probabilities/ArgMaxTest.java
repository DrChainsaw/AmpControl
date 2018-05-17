package ampcontrol.amp.probabilities;

import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link ArgMax}
 *
 * @author Christian Skärby
 */
public class ArgMaxTest {

    /**
     * Test that the right element is selected. Takes very long time due to Nd4j long startup
     *
     */
    @Test
    public void apply() {
        final Interpreter<Integer> argMax = new ArgMax();
        assertEquals("Incorrect output", Collections.singletonList(2), argMax.apply(Nd4j.create(new double[] {0 , -10.5, 2.777, 2.776})));
    }

}