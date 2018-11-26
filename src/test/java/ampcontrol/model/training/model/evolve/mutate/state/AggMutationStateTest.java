package ampcontrol.model.training.model.evolve.mutate.state;

import org.apache.commons.lang.mutable.MutableInt;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link AggMutationState}
 *
 * @author Christian Skärby
 */
public class AggMutationStateTest {

    /**
     * Test the builder
     */
    @Test
    public void mutate() {
        final MutableInt state0 = new MutableInt(0);
        final MutableInt state1 = new MutableInt(0);
        final MutableInt state2 = new MutableInt(0);
        final MutationState<String, MutableInt> aggMutationState = AggMutationState.<String, MutableInt>builder()
                .first(new GenericMutationState<>(state -> str -> str + "_" + state.intValue() + state0.intValue()))
                .second(new GenericMutationState<>(state -> str -> str + "_" + state.intValue() + state1.intValue()))
                .andThen(new GenericMutationState<>(state -> str -> str + "_" + state.intValue() + state2.intValue()))
                .build();
        state0.setValue(1);
        state1.setValue(2);
        state2.setValue(3);
        assertEquals("Incorrect output!", "mutation_01_02_03", aggMutationState.mutate("mutation", new MutableInt(0)));
    }
}