package ampcontrol.model.training.model.evolve.crossover.state;

import ampcontrol.model.training.model.evolve.crossover.Crossover;

import java.util.function.Function;

/**
 * {@link CrossoverState} where state is of a generic type.
 * @param <T>
 * @param <S>
 *
 * @author Christian Skärby
 */
public class GenericCrossoverState<T, S> implements CrossoverState<T,S> {

    private final MergeFunction<T, S> mergeFunction;
    private final Function<S, Crossover<T>> factory;


    public interface MergeFunction<T, S> {
        void merge(S thisState, S otherState, T thisInput, T otherInput, T result);
    }

    public GenericCrossoverState(MergeFunction<T, S> mergeFunction,
                                 Function<S, Crossover<T>> factory) {
        this.mergeFunction = mergeFunction;
        this.factory = factory;
    }

    @Override
    public T cross(T first, T second, S stateFirst, S stateSecond) {
        T result = factory.apply(stateFirst).cross(first,second);
        mergeFunction.merge(stateFirst, stateSecond, first, second, result);
        return result;
    }
}
