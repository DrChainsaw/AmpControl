package ampcontrol.model.training.model.evolve.mutate;

import java.util.stream.Stream;

/**
 * Interface for mutation operation.
 *
 * @param <T> Type to mutate
 * @author Christian Skärby
 */
public interface Mutation<T> {

    /**
     * Interface for supplying mutations. Motivation is to allow for serialization of mutation info as it might not
     * always be possible to create it from scratch.
     *
     * @param <T>
     */
    interface Supplier<T> {
        Stream<T> stream();
    }

    /**
     * Applies mutation to the provided input
     *
     * @param toMutate The builder to mutate
     * @return The mutated builder. Note: might not be same instance as input!
     */
    T mutate(T toMutate);
}
