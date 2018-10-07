package ampcontrol.model.training.model.evolve.mutate.state;


import ampcontrol.model.training.model.evolve.mutate.Mutation;

/**
 * Wrapper for {@link Mutation}s which does not require any state to be copied or persisted.
 * @param <T>
 *
 * @author Christian Skärby
 */
public class NoMutationStateWapper<T> implements MutationState<T> {

    private final Mutation<T> mutation;

    public NoMutationStateWapper(Mutation<T> mutation) {
        this.mutation = mutation;
    }

    @Override
    public T mutate(T toMutate) {
        return mutation.mutate(toMutate);
    }

    @Override
    public void save(String baseName) {
        // No state to save
    }

    @Override
    public MutationState<T> clone() {
        // No state to clone
        return this;
    }
}
