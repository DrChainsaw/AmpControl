package ampcontrol.amp.labelmapping;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link MaskDuplicateLabelMapping}
 *
 * @author Christian Skärby
 */
public class MaskDuplicateLabelMappingTest {

    private final static LabelMapping<Integer> mockMap = MockMap.getPositive();
    private final static LabelMapping<Integer> negaMap = MockMap.getNegative();

    /**
     * Test that mask is applied correctly
     */
    @Test
    public void applyMask() {
        LabelMapping<Integer> mask = new MaskDuplicateLabelMapping<>(mockMap);
        final int test1 = 1;
        final int test2 = 2;
        assertEquals("Masked non duplicate!", mockMap.apply(test1), mask.apply(test1));
        assertEquals("Did not mask duplicate!", negaMap.apply(test1), mask.apply(test1));
        assertEquals("Masked non duplicate!", mockMap.apply(test2), mask.apply(test2));
        assertEquals("Did not mask duplicate!", negaMap.apply(test2), mask.apply(test2));
    }
}