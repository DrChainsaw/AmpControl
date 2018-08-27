package ampcontrol.model.training.data.state;

import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.*;

/**
 * Test cases for {@link ResetableStateFactory}
 *
 * @author Christian Skärby
 */
public class ResetableStateFactoryTest {


    /**
     * Tests that created {@link Random} has right seed and can be reseted
     */
    @Test
    public void testRandomCreation() {
        final long seed = 666;
        final Random ref = new Random(seed);
        final ResetableStateFactory factory = new ResetableStateFactory(seed);
        final Random testRandom = factory.createNewRandom();

        final int[] refSeq = createSequence(ref);

        assertArrayEquals("Incorrect sequence!", refSeq, createSequence(testRandom));

        factory.restorePreviousState();
        assertArrayEquals("Incorrect sequence!", refSeq, createSequence(testRandom));

        final Random newTestRandom = factory.createNewRandom();
        assertNotEquals("Incorrect sequence!", Arrays.toString(refSeq), Arrays.toString(createSequence(newTestRandom)));
    }

    /**
     * Test that a {@link HashMap} can be managed correctly
     */
    @Test
    public void testStateReference() {
        final Map<Integer, String> initMap = new HashMap<>();
        initMap.put(7, "test7");
        initMap.put(2, "test2");
        final ResetableStateFactory factory = new ResetableStateFactory(123);

        Supplier<Map<Integer, String>> testSupplier = factory.createNewStateReference(HashMap::new, initMap);
        assertEquals("Incorrect state!", initMap, testSupplier.get());
        testSupplier.get().put(6, "test6");

        assertTrue("Added element expected to be present!", testSupplier.get().containsKey(6));

        factory.restorePreviousState();
        assertFalse("Element was not part of previous state!", testSupplier.get().containsKey(6));
        assertTrue("Element was part of previous state!", testSupplier.get().containsKey(7));
        assertTrue("Element was part of previous state!", testSupplier.get().containsKey(2));
    }


    private int[] createSequence(Random r) {
        return r.ints(5).toArray();
    }
}