package ampcontrol.amp.labelmapping;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link MaskingLabelMapping}
 *
 * @author Christian Skärby
 */
public class MaskingLabelMappingTest {

    private final static LabelMapping<Integer> mockMap = MockMap.getPositive();
    private final static LabelMapping<Integer> negaMap = MockMap.getNegative();

    /**
     * Test that masking is applied
     */
    @Test
    public void applyMask() {
        final int expectMask = 666;
        final LabelMapping<Integer> mask = new MaskingLabelMapping<>(Arrays.asList(expectMask), mockMap);
        assertEquals("Did not mask!", negaMap.apply(expectMask), mask.apply(expectMask));
    }

    /**
     * Test that masking is not applied
     */
    @Test
    public void doNotApplyMask() {
        final int expectNoMask = 666;
        final LabelMapping<Integer> mask = new MaskingLabelMapping<>(Arrays.asList(expectNoMask - 1), mockMap);
        assertEquals("Did not mask!", mockMap.apply(expectNoMask), mask.apply(expectNoMask));
    }
}