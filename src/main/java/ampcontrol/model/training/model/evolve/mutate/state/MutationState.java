package ampcontrol.model.training.model.evolve.mutate.state;

/**
 * Interface for mutation operation based on some state which might also be mutated. Typical use case is that state
 * describes how mutation may be carried out and that this also changes depending on outcome of the mutation.
 * @param <T>
 * @param <S>
 *
 * @author Christian Skärby
 */
public interface MutationState<T, S> {

    /**
     * Applies mutation to the provided item
     *
     * @param toMutate The item to mutate
     * @param state State which may be used to perform the mutation. May also be mutated as a result
     * @return The mutated item. Note: might not be same instance as input!
     */
    T mutate(T toMutate, S state);

}
