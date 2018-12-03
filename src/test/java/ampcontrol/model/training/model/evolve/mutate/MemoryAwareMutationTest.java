package ampcontrol.model.training.model.evolve.mutate;

import org.junit.Test;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertFalse;

/**
 * Test cases for {@link MemoryAwareMutation}
 *
 * @author Christian Skärby
 */
public class MemoryAwareMutationTest {

    /**
     * Test that mutation works as expected. Class does not do much by itself so test is a bit trivial. Testing with
     * actual device usage does not seem feasible.
     */
    @Test
    public void mutate() {
       assertTrue("Incorrect mutation used!", new MemoryAwareMutation<Boolean>(() -> 0.7, usage -> {
           assertEquals("Did not get usage!", 0.7, usage, 1e-10);
           return toMut -> true;
       }).mutate(false));

        assertFalse("Incorrect mutation used!", new MemoryAwareMutation<Boolean>(() -> 0.2, usage -> {
            assertEquals("Did not get usage!", 0.2, usage, 1e-10);
            return toMut -> false;
        }).mutate(true));
    }
}