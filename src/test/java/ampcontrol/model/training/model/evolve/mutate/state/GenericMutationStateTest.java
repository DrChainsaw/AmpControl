package ampcontrol.model.training.model.evolve.mutate.state;

import ampcontrol.model.training.model.evolve.mutate.Mutation;
import ampcontrol.model.training.model.evolve.state.GenericState;
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
        final MutableInt state = new MutableInt(13);
        final Mutation<String> mutationState = createIntStateMutation(state);
        assertEquals("Incorrect output!", "mutate_13", mutationState.mutate("mutate"));
    }

    @Test
    public void mutateClone() {
        final MutableInt state = new MutableInt(0);
        final Mutation<String> mutationState = createIntStateMutation(state).clone();
        state.setValue(666);
        assertEquals("Incorrect output!", "mutate_0", mutationState.mutate("mutate"));
    }


    static MutationState<String> createIntStateMutation(MutableInt startState) {
        return new GenericMutationState<>(
                new GenericState<>(
                        startState,
                        mutableInt -> new MutableInt(mutableInt.intValue()),
                        (str, state) -> {/* Ignore*/}),
                state -> str -> str + "_" + state.intValue());
    }

}