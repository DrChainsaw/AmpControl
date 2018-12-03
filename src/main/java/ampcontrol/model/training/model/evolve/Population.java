package ampcontrol.model.training.model.evolve;

import java.util.stream.Stream;

/**
 * Represens a population. Callers must assume that population might change over time.
 *
 * @author Christian Skärby
 */
public interface Population<T> {

    /**
     * Returns a {@link Stream} of the population
     * @return a {@link Stream}
     */
    Stream<T> streamPopulation();

    /**
     * Add a Runnable which will be invoked if the population changes
     * @param callback will be invoked if population changes
     */
    void onChangeCallback(Runnable callback);
}
