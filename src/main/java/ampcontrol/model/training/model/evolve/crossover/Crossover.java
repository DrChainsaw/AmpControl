package ampcontrol.model.training.model.evolve.crossover;

/**
 * Interface for crossover operation.
 * @param <T>
 * @author Christian Skärby
 */
public interface Crossover<T> {


    /**
     * Return the crossover between the two given items.
     * @param first First item
     * @param second Second item
     * @return The result of the crossover.
     */
    T cross(T first, T second);
}
