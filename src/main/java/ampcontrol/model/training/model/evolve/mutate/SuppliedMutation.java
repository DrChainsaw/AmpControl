package ampcontrol.model.training.model.evolve.mutate;

/**
 * Applies a mutation provided by a supplier. Main use case is to be able to create {@link Mutation}s after some
 * other mutation has modified state which is needed to initialize the mutation. Example is if layers are removed
 * before other mutations are applied.
 * @param <T>
 *
 * @author Christian Skärby
 */
public class SuppliedMutation<T> implements Mutation<T> {

    private final java.util.function.Supplier<Mutation<T>> supplier;

    public SuppliedMutation(java.util.function.Supplier<Mutation<T>> supplier) {
        this.supplier = supplier;
    }


    @Override
    public T mutate(T toMutate) {
        return supplier.get().mutate(toMutate);
    }
}
