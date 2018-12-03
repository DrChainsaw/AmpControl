package ampcontrol.model.training.model.evolve.state;

import java.io.IOException;

/**
 * Representation of some state which may be cloned and persisted.
 * @param <T>
 *
 * @author Christian Skärby
 */
public interface PersistentState<T> {

    /**
     * Save the current state.
     *
     * @param baseName Base name. Implementations are expected to append to it in order to guarantee uniqueness
     */
    void save(String baseName) throws IOException;

    /**
     * Create an independent copy of the state. Changes in this instance will not be reflected in the clone and vice versa
     * @return an independent copy of the state
     */
    PersistentState<T> clone();

}
