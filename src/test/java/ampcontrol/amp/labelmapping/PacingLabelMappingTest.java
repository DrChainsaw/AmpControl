package ampcontrol.amp.labelmapping;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Test cases for {@link PacingLabelMapping}
 *
 * @author Christian Skärby
 */
public class PacingLabelMappingTest {

    private final static LabelMapping<Integer> mockMap = MockMap.getPositive();
    private final static LabelMapping<Integer> negaMap = MockMap.getNegative();

    private final static int paceMs = 5;

    /**
     * Test pacing applied
     */
    @Test
    public void apply() {
        final int label = 666;
        LabelMapping<Integer> paceMap = new PacingLabelMapping<>(paceMs, mockMap);
        assertEquals("Pacing applied! ", mockMap.apply(label), paceMap.apply(label));
        assertEquals("Pacing not applied! ", negaMap.apply(label), paceMap.apply(label));

        try {
            Thread.sleep(paceMs * 2);
            assertEquals("Pacing applied! ", mockMap.apply(label), paceMap.apply(label));
            assertEquals("Pacing not applied! ", negaMap.apply(label), paceMap.apply(label));

        } catch (InterruptedException e) {
            fail("Test interrupted");
        }
    }
}