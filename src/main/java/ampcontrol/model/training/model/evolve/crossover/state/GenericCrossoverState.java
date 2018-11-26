package ampcontrol.model.training.model.evolve.crossover.state;

import ampcontrol.model.training.model.evolve.crossover.Crossover;
import ampcontrol.model.training.model.evolve.state.AccessibleState;

import java.io.IOException;
import java.util.function.Function;

/**
 * {@link CrossoverState} where state is of a generic type.
 * @param <T>
 * @param <V>
 *
 * @author Christian Skärby
 */
public class GenericCrossoverState<T, V> extends CrossoverState<T,V> {

    private final AccessibleState<V> state;
    private final MergeFunction<T, V> mergeFunction;
    private final Function<V, Crossover<T>> factory;


    public interface MergeFunction<T, V> {
        void merge(V thisState, V otherState, T thisInput, T otherInput, T result);
    }

    public GenericCrossoverState(AccessibleState<V> state,
                                 MergeFunction<T, V> mergeFunction,
                                 Function<V, Crossover<T>> factory) {
        this.state = state;
        this.mergeFunction = mergeFunction;
        this.factory = factory;
    }

    @Override
    public void save(String baseName) throws IOException {
        state.save(baseName);
    }

    @Override
    public GenericCrossoverState<T, V> clone() {
        return new GenericCrossoverState<>(state.clone(), mergeFunction, factory);
    }

    @Override
    public void merge(CrossoverState<T, V> other , T thisInput, T otherInput, T result) {
        mergeFunction.merge(state.get(), other.getState(), thisInput, otherInput, result);
    }

    @Override
    protected V getState() {
        return state.get();
    }

    @Override
    public T cross(T first, T second) {
        return factory.apply(state.get()).cross(first,second);
    }
}
