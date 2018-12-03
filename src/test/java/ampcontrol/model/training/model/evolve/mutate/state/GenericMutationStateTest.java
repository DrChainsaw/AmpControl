package ampcontrol.model.training.model.evolve.mutate.state;

import org.apache.commons.lang.mutable.MutableInt;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link GenericMutationState}
 *
 * @author Christian Skärby
 */
public class GenericMutationStateTest {

    /**
     * Test mutate function
     */
    @Test
    public void mutate() {
        final MutationState<String, MutableInt> mutationState = new GenericMutationState<>(state -> str -> str + "_" + state.intValue());
        final MutableInt state = new MutableInt(13);
        assertEquals("Incorrect output!", "mutate_13", mutationState.mutate("mutate", state));
    }

}