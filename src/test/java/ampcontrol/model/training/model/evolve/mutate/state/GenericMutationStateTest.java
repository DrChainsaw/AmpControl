package ampcontrol.model.training.model.evolve.mutate.state;

import ampcontrol.model.training.model.evolve.mutate.Mutation;
import lombok.Getter;
import org.apache.commons.lang.mutable.MutableInt;
import org.junit.Assert;
import org.junit.Test;

import java.util.function.BiConsumer;
import java.util.function.UnaryOperator;

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
        final Mutation<String> mutationState =  createIntStateMutation(state);
        assertEquals("Incorrect output!", "mutate_13", mutationState.mutate("mutate"));
    }

    @Test
    public void mutateClone() {
        final MutableInt state = new MutableInt(0);
        final Mutation<String> mutationState =  createIntStateMutation(state).clone();
        state.setValue(666);
        assertEquals("Incorrect output!", "mutate_0", mutationState.mutate("mutate"));
    }


    @Test
    public void save() {
        final SaveProbe probe = new SaveProbe(17);
        final MutationState<Integer> mutationState = createSaveStateMutation(17, probe);
        mutationState.save("test");
        assertEquals("Incorrect save str", "test", probe.getLastString());
    }

    static MutationState<String> createIntStateMutation(MutableInt startState) {
        return new GenericMutationState<>(
                startState,
                mutableInt -> new MutableInt(mutableInt.intValue()),
                (str, state) -> {/* Ignore*/},
                state -> str -> str + "_" + state.intValue());
    }

    static MutationState<Integer> createSaveStateMutation(int startState, BiConsumer<String, Integer> saveOperation) {
        return new GenericMutationState<>(
                startState,
                UnaryOperator.identity(),
                saveOperation,
                state -> intInput -> intInput + state
        );
    }

    @Getter
    static final class SaveProbe implements BiConsumer<String, Integer> {

        private String lastString = "NONE";
        private final int expectInt;

        SaveProbe(int expectInt) {
            this.expectInt = expectInt;
        }

        @Override
        public void accept(String s, Integer integer) {
            lastString = s;
            Assert.assertEquals("Incorrect state to save!", expectInt, integer.intValue());
        }
    }

}