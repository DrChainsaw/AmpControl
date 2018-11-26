package ampcontrol.model.training.model.evolve.crossover.state;


/**
 * Interface for crossover operation based on some state which might also be mutated. Typical use case is that state
 * describes how crossover may be carried out and that this also changes depending on outcome of the mutation.
 * @param <T>
 * @param <S>
 *
 * @author Christian Skärby
 */
public interface CrossoverState<T, S> {

    /**
     * Return the crossover between the two given items
     * @param first First item
     * @param second Second item
     * @param stateFirst state for first item
     * @param stateSecond state for second item
     * @return The result of the crossover.
     */
    T cross(T first, T second, S stateFirst, S stateSecond);
}
