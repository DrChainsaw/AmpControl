package ampcontrol.amp.labelmapping;

import java.util.Collections;

/**
 * MockMaps for testing
 */
public class MockMap {
    private final static LabelMapping<Integer> mockMapPostive = lab -> Collections.singletonList(lab);
    private final static LabelMapping<Integer> mockMapNegative = lab -> Collections.emptyList();

    /**
     * Returns an instance which always returns the input
     *
     * @return
     */
    static LabelMapping<Integer> getPositive() {
        return mockMapPostive;
    }

    /**
     * Returns an instance which always returns no output
     *
     * @return
     */
    static LabelMapping<Integer> getNegative() {
        return mockMapNegative;
    }

}
