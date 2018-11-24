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
public interface CrossoverState<T, V extends CrossoverState<T,V>> extends Crossover<T>, PersistentState<V> {

    /**
     * Merge this CrossoverState with another CrossoverState
     * @param other CrossoverState to merge with
     * @param thisInput Input used for this crossover
     * @param otherInput Input used for other
     * @param result Result of crossover
     * @return The result of the merge
     */
    V merge(V other, T thisInput, T otherInput, T result);

    /**
     * "Fake" override of PersistenState#clone
     * @return a clone
     */
    CrossoverState<T, V> clone();

}
