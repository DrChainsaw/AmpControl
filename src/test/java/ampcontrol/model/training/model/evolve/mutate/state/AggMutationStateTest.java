package ampcontrol.model.training.model.evolve.mutate.state;

import ampcontrol.model.training.model.evolve.state.GenericState;
import lombok.Getter;
import org.apache.commons.lang.mutable.MutableInt;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.util.List;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static ampcontrol.model.training.model.evolve.mutate.state.GenericMutationStateTest.createIntStateMutation;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

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
        final MutationState<String> aggMutationState = AggMutationState.<String>builder()
                .andThen(createIntStateMutation(state0))
                .andThen(createIntStateMutation(state1))
                .andThen(createIntStateMutation(state2))
                .build();
        state0.setValue(1);
        state1.setValue(2);
        state2.setValue(3);
        assertEquals("Incorrect output!", "mutation_1_2_3", aggMutationState.mutate("mutation"));
    }

    /**
     * Test that a cloned instance retains its old state.
     */
    @Test
    public void mutateClone() {
        final MutableInt state0 = new MutableInt(0);
        final MutableInt state1 = new MutableInt(0);
        final MutableInt state2 = new MutableInt(0);
        final MutableInt state3 = new MutableInt(0);
        final MutationState<String> aggMutationState = AggMutationState.<String>builder()
                .first(createIntStateMutation(state0))
                .second(createIntStateMutation(state1))
                .andThen(createIntStateMutation(state2))
                .andThen(createIntStateMutation(state3))
                .build();
        state0.setValue(6);
        state1.setValue(7);
        state2.setValue(8);
        state3.setValue(10);
        final MutationState<String> cloned = aggMutationState.clone();
        state0.setValue(0);
        state1.setValue(2);
        state2.setValue(9);
        state2.setValue(17);
        assertEquals("Incorrect output!", "mutation_6_7_8_10", cloned.mutate("mutation"));
    }

    /**
     * Test that the save method propagates
     */
    @Test
    public void save() {
        final List<SaveProbe> probes = IntStream.range(0, 17).mapToObj(SaveProbe::new).collect(Collectors.toList());
        final AggMutationState<Integer> aggMutationState = probes.stream()
                .reduce(AggMutationState.<Integer>builder(),
                        (builder, probe) -> builder.andThen(createSaveStateMutation(probe.getExpectInt(), probe)),
                        (builder1, builder2) -> builder2)
                .build();

        try {
            aggMutationState.save("test");
            probes.forEach(probe -> assertEquals("Incorrect dirname!", "test", probe.getLastString()));
        } catch (IOException e) {
            fail("Unexpected exception!");
        }
    }

    private static MutationState<Integer> createSaveStateMutation(int startState, GenericState.PersistState<Integer> saveOperation) {
        return new GenericMutationState<>(
                new GenericState<>(
                        startState,
                        UnaryOperator.identity(),
                        saveOperation),
                state -> intInput -> intInput + state
        );
    }

    @Getter
    private static final class SaveProbe implements GenericState.PersistState<Integer> {

        private String lastString = "NONE";
        private final int expectInt;

        SaveProbe(int expectInt) {
            this.expectInt = expectInt;
        }

        @Override
        public void save(String s, Integer integer) {
            lastString = s;
            Assert.assertEquals("Incorrect state to save!", expectInt, integer.intValue());
        }
    }

}