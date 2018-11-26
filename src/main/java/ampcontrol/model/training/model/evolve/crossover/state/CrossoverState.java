package ampcontrol.model.training.model.evolve.crossover.state;

import ampcontrol.model.training.model.evolve.crossover.Crossover;
import ampcontrol.model.training.model.evolve.state.PersistentState;

/**
 * {@link Crossover} which (maybe) has state which is mutated as well. Apart from the capabilities of PersistentState,
 * it is also able to merge with another {@link CrossoverState} of the same kind.
 * @param <T>
 *
 * @author Christian Skärby
 */
public abstract class CrossoverState<T, V> implements Crossover<T>, PersistentState<V> {

    /**
     * Merge this CrossoverState with another CrossoverState
     * @param other CrossoverState to merge with
     * @param thisInput Input used for this crossover
     * @param otherInput Input used for other
     * @param result Result of crossover
     */
    public abstract void merge(CrossoverState<T,V> other, T thisInput, T otherInput, T result);

    /**
     * "Fake" override of PersistentState#clone
     * @return a clone
     */
    public abstract CrossoverState<T, V> clone();

    /**
     * Returns the state
     * @return the state
     */
    protected abstract V getState();
}
