package ampcontrol.model.training.model.evolve.crossover.state;

import ampcontrol.model.training.model.evolve.crossover.Crossover;

/**
 * Wrapper for {@link Crossover}s which does not require any state.
 * @param <T>
 * @param <S>
 * @author Christian Skärby
 */
public class NoCrossoverStateWapper<T, S> implements CrossoverState<T, S> {

    private final Crossover<T> crossover;

    public NoCrossoverStateWapper(Crossover<T> crossover) {
        this.crossover = crossover;
    }

    @Override
    public T cross(T first, T second, S firsState, S secondState) {
        return crossover.cross(first, second);
    }
}
