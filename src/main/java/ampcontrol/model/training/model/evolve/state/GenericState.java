package ampcontrol.model.training.model.evolve.state;

import java.io.IOException;
import java.util.function.UnaryOperator;

/**
 * {@link PersistentState} where state is of generic type, copied through a provided {@link UnaryOperator} and
 * persisted through a {@link PersistentState}.
 * @param <T>
 *
 * @author Christian Skärby
 */
public class GenericState<T> implements AccessibleState<T> {

    private final T state;
    private final UnaryOperator<T> copyState;
    private final PersistState<T> persistState;

    /**
     * Interface for persisting state
     */
    public interface PersistState<V> {
        void save(String baseName, V state) throws IOException;
    }

    public GenericState(T state, UnaryOperator<T> copyState, PersistState<T> persistState) {
        this.state = state;
        this.copyState = copyState;
        this.persistState = persistState;
    }

    @Override
    public T get() {
        return state;
    }

    @Override
    public void save(String baseName) throws IOException {
        persistState.save(baseName, state);
    }

    @Override
    public GenericState<T> clone() {
        return new GenericState<>(copyState.apply(state), copyState, persistState);
    }
}
