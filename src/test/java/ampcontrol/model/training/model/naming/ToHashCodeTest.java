package ampcontrol.model.training.model.naming;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link ToHashCode}
 *
 * @author Christian Skärby
 */
public class ToHashCodeTest {

    /**
     * Test that the hashcode is returned as a String
     */
    @Test
    public void toFileName() {
        assertEquals("Incorrect name!", String.valueOf("abcd".hashCode()), FileNamePolicy.toHashCode.toFileName("abcd"));
    }
}