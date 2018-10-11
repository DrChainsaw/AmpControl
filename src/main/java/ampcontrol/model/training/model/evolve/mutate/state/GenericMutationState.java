package ampcontrol.model.training.model.evolve.mutate.state;

import ampcontrol.model.training.model.evolve.mutate.Mutation;

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

    private final V state;
    private final UnaryOperator<V> copyState;
    private final PersistState<V> persistState;
    private final Function<V, Mutation<T>> factory;

    /**
     * Interface for persisting state
     */
    public interface PersistState<V> {
        void save(String baseName, V state) throws IOException;
    }

    public GenericMutationState(V state, UnaryOperator<V> copyState, PersistState<V> persistState, Function<V, Mutation<T>> factory) {
        this.state = state;
        this.copyState = copyState;
        this.persistState = persistState;
        this.factory = factory;
    }

    @Override
    public void save(String baseName) throws IOException {
        persistState.save(baseName, state);
    }

    @Override
    public MutationState<T> clone() {
        return new GenericMutationState<>(copyState.apply(state), copyState, persistState, factory);
    }

    @Override
    public T mutate(T toMutate) {
        return factory.apply(state).mutate(toMutate);
    }
}
