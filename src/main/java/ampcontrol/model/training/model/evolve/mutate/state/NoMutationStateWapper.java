package ampcontrol.model.training.model.evolve.mutate.state;


import ampcontrol.model.training.model.evolve.mutate.Mutation;

/**
 * Wrapper for {@link Mutation}s which does not require any state.
 * @param <T>
 * @param <S>
 * @author Christian Skärby
 */
public class NoMutationStateWapper<T,S> implements MutationState<T,S> {

    private final Mutation<T> mutation;

    public NoMutationStateWapper(Mutation<T> mutation) {
        this.mutation = mutation;
    }

    @Override
    public T mutate(T toMutate, S state) {
        return mutation.mutate(toMutate);
    }
}
