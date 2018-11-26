package ampcontrol.model.training.model.evolve.mutate.state;

import ampcontrol.model.training.model.evolve.mutate.Mutation;
import ampcontrol.model.training.model.evolve.state.AccessibleState;

import java.io.IOException;
import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * {@link MutationState} where state is of generic type, copied through a provided {@link UnaryOperator} and
 * transformed to a {@link Mutation} though a factory {@link Function}.
 * @param <T>
 * @param <V>
 *
 * @author Christian Skärby
 */
public class GenericMutationState<T,V> implements MutationState<T> {

    private final AccessibleState<V> state;
    private final Function<V, Mutation<T>> factory;

    public GenericMutationState(AccessibleState<V> state, Function<V, Mutation<T>> factory) {
        this.state = state;
        this.factory = factory;
    }

    @Override
    public void save(String baseName) throws IOException {
        state.save(baseName);
    }

    @Override
    public MutationState<T> clone() {
        return new GenericMutationState<>(state.clone(), factory);
    }

    @Override
    public T mutate(T toMutate) {
        return factory.apply(state.get()).mutate(toMutate);
    }
}
