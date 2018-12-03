package ampcontrol.model.training.model;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.crossover.graph.GraphInfo;
import ampcontrol.model.training.model.evolve.crossover.state.GenericCrossoverState;
import ampcontrol.model.training.model.evolve.mutate.state.GenericMutationState;
import ampcontrol.model.training.model.evolve.state.AccessibleState;
import ampcontrol.model.training.model.evolve.state.GenericState;
import ampcontrol.model.training.model.evolve.state.SharedSynchronizedState;
import org.apache.commons.lang3.mutable.Mutable;
import org.apache.commons.lang3.mutable.MutableObject;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.junit.Test;

import java.util.function.UnaryOperator;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Test cases for {@link EvolvingGraphAdapter}
 *
 * @author Christian Skärby
 */
public class EvolvingGraphAdapterTest {

    private final static UnaryOperator<SharedSynchronizedState.View<Mutable<String>>> copyState =
            view -> view.copy().update(new MutableObject<>(view.get().getValue()));

    /**
     * Test that a model can be evolved
     */
    @Test
    public void evolve() {

        final AccessibleState<SharedSynchronizedState.View<Mutable<String>>> state = new GenericState<>(
                new SharedSynchronizedState<Mutable<String>>(new MutableObject<>("initial")).view(),
                copyState,
                (str, currState) -> fail("Should not happen!"));

        final Mutable<String> stateSpy = new MutableObject<>();
        final EvolvingGraphAdapter<SharedSynchronizedState.View<Mutable<String>>> graphAdapter = createGraphAdapter(state, stateSpy);

        final EvolvingGraphAdapter<SharedSynchronizedState.View<Mutable<String>>> evolvedGraphAdapter = graphAdapter.evolve();
        assertEquals("Shall not modify state of original instance!", "initial", state.get().get().getValue());
        assertEquals("Shall modify state of new instance!", "initial_mutated", stateSpy.getValue());
        assertEquals("Incorrect weights!", graphAdapter.asModel().params(), evolvedGraphAdapter.asModel().params());
    }

    @Test
    public void cross() {
        final AccessibleState<SharedSynchronizedState.View<Mutable<String>>> state = new GenericState<>(
                new SharedSynchronizedState<Mutable<String>>(new MutableObject<>("initial")).view(),
                copyState,
                (str, currState) -> fail("Should not happen!"));

        final Mutable<String> stateSpy = new MutableObject<>();
        final EvolvingGraphAdapter<SharedSynchronizedState.View<Mutable<String>>> graphAdapter1 = createGraphAdapter(state, stateSpy);
        final EvolvingGraphAdapter<SharedSynchronizedState.View<Mutable<String>>> graphAdapter2 = createGraphAdapter(state, stateSpy);
        final EvolvingGraphAdapter<SharedSynchronizedState.View<Mutable<String>>> crossBreed = graphAdapter1.cross(graphAdapter2);

        assertEquals("Shall not modify state of original instance!", "initial", state.get().get().getValue());
        assertEquals("Shall modify state of new instance!", "initial_cross_initial", stateSpy.getValue());
        assertEquals("Incorrect weights!", graphAdapter1.asModel().params(), crossBreed.asModel().params());
    }

    @Test
    public void evolveAndCross() {
        final AccessibleState<SharedSynchronizedState.View<Mutable<String>>> state = new GenericState<>(
                new SharedSynchronizedState<Mutable<String>>(new MutableObject<>("initial")).view(),
                copyState,
                (str, currState) -> fail("Should not happen!"));

        final Mutable<String> stateSpy = new MutableObject<>();
        final EvolvingGraphAdapter<SharedSynchronizedState.View<Mutable<String>>> graphAdapter1 = createGraphAdapter(state, stateSpy);
        final EvolvingGraphAdapter<SharedSynchronizedState.View<Mutable<String>>> graphAdapter2 = graphAdapter1.evolve();

        assertEquals("Shall not modify state of original instance!", "initial", state.get().get().getValue());
        assertEquals("Shall modify state of new instance!","initial_mutated", stateSpy.getValue());

        final EvolvingGraphAdapter<SharedSynchronizedState.View<Mutable<String>>> crossBreed = graphAdapter1.cross(graphAdapter2);

        assertEquals("Shall not modify state of original instance!", "initial", state.get().get().getValue());
        assertEquals("Shall modify state of new instance!","initial_cross_initial_mutated", stateSpy.getValue());
        assertEquals("Incorrect weights!", graphAdapter1.asModel().params(), crossBreed.asModel().params());
    }

    private EvolvingGraphAdapter<SharedSynchronizedState.View<Mutable<String>>> createGraphAdapter(
            AccessibleState<SharedSynchronizedState.View<Mutable<String>>> state,
            Mutable<String> stateSpy) {

        return EvolvingGraphAdapter.<SharedSynchronizedState.View<Mutable<String>>>builder(GraphUtils.getGraph("1", "2", "3"))
                .evolutionState(state)
                .mutation(new GenericMutationState<>(
                        currState -> {
                            currState.get().setValue(currState.get().getValue() + "_mutated");
                            stateSpy.setValue(currState.get().getValue());
                            return builder -> builder;
                        }
                ))
                .crossover(new GenericCrossoverState<>(
                        (stateThis, stateOther, a, b, c) -> {
                            stateThis.get().setValue(stateThis.get().getValue() + "_cross_" + stateOther.get().getValue());
                            stateSpy.setValue(stateThis.get().getValue());
                        },
                        str -> (info1, info2) -> new GraphInfo() {

                            @Override
                            public ComputationGraphConfiguration.GraphBuilder builder() {
                                return info1.builder();
                            }

                            @Override
                            public Stream<NameMapping> verticesFrom(GraphInfo info) {
                                if(info == info1) {
                                    return info1.verticesFrom(info1);
                                }
                                return Stream.empty();
                            }
                        }
                ))
                .build();
    }
}