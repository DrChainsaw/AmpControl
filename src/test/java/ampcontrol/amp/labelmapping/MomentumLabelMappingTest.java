package ampcontrol.amp.labelmapping;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link MomentumLabelMapping}
 *
 * @author Christian Skärby
 */
public class MomentumLabelMappingTest {

    private final static LabelMapping<Integer> mockMap = MockMap.getPositive();
    private final static LabelMapping<Integer> negaMap = MockMap.getNegative();

    /**
     * Test that momentum filter is applied
     */
    @Test
    public void apply() {
        final int momentumThreshold = 3;
        LabelMapping<Integer> momFilt = new MomentumLabelMapping<>(momentumThreshold, mockMap);
        final int label0 = 0;
        final int label1 = label0 + 1;
        assertEquals("Filtering not applied!", negaMap.apply(label0), momFilt.apply(label0));
        assertEquals("Filtering not applied!", negaMap.apply(label0), momFilt.apply(label0));
        // Change to label1
        assertEquals("Filtering not applied!", negaMap.apply(label1), momFilt.apply(label1));
        assertEquals("Filtering not applied!", negaMap.apply(label1), momFilt.apply(label1));
        assertEquals("Filtering applied!", mockMap.apply(label1), momFilt.apply(label1));
        // And back to label0
        assertEquals("Filtering not applied!", negaMap.apply(label0), momFilt.apply(label0));
        assertEquals("Filtering not applied!", negaMap.apply(label0), momFilt.apply(label0));
        assertEquals("Filtering applied!", mockMap.apply(label0), momFilt.apply(label0));
        assertEquals("Filtering applied!", mockMap.apply(label0), momFilt.apply(label0));
    }
}