package ampcontrol.model.training.model.evolve.mutate.state;

import ampcontrol.model.training.model.evolve.mutate.Mutation;

import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * {@link MutationState} where state is of generic type, copied through a provided {@link UnaryOperator} and
 * transformed to a {@link Mutation} though a factory {@link Function}.
 * @param <T>
 * @param <S>
 *
 * @author Christian Skärby
 */
public class GenericMutationState<T,S> implements MutationState<T, S> {

    private final Function<S, Mutation<T>> factory;

    public GenericMutationState(Function<S, Mutation<T>> factory) {
        this.factory = factory;
    }

    @Override
    public T mutate(T toMutate, S state) {
        return factory.apply(state).mutate(toMutate);
    }

}
