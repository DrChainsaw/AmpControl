package ampcontrol.model.training.model.evolve.mutate.state;

import ampcontrol.model.training.model.evolve.mutate.Mutation;

import java.io.IOException;

/**
 * {@link Mutation} which (maybe) has state which is mutated as well. Shall be able to duplicate its state to allow for
 * several different offspring from the same state. State may also be persisted.
 * @param <T>
 *
 * @author Christian Skärby
 */
public interface MutationState<T> extends Mutation<T> {

    /**
     * Save the current state.
     *
     * @param baseName Base name. Implementations are expected to append to it in order to guarantee uniqueness
     */
    void save(String baseName) throws IOException;

    /**
     * Create a clone
     * @return a clone
     */
    MutationState<T> clone();

}
