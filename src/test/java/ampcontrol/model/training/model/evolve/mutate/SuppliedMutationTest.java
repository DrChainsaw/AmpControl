package ampcontrol.model.training.model.evolve.mutate;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link SuppliedMutation}
 *
 * @author Christian Skärby
 */
public class SuppliedMutationTest {

    /**
     * Test mutation function. Trivial test due to trivial class
     */
    @Test
    public void mutate() {
        assertEquals("Incorrect output!", "testStr_mutated",
                new SuppliedMutation<String>(() -> str -> str + "_mutated").mutate("testStr"));
    }
}