package ampcontrol.model.training.model.naming;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link CharThreshold}
 *
 * @author Christian Skärby
 */
public class CharThresholdTest {

    /**
     * Test that the right policy is selected
     */
    @Test
    public void toFileName() {
        final String aboveStr = "above";
        final FileNamePolicy policy = new CharThreshold(5, str -> aboveStr);
        assertEquals("Incorrect name!", "abcde", policy.toFileName("abcde"));
        assertEquals("Incorrect name!", aboveStr, policy.toFileName("abcdefg"));
    }

    /**
     * Test that the right policy is selected
     */
    @Test
    public void toFileNameWithBelowPolicy() {
        final String aboveStr = "above";
        final String belowStr = "below";
        final FileNamePolicy policy = new CharThreshold(5, str -> aboveStr, str->belowStr);
        assertEquals("Incorrect name!", belowStr, policy.toFileName("abcde"));
        assertEquals("Incorrect name!", aboveStr, policy.toFileName("abcdefg"));
    }
}