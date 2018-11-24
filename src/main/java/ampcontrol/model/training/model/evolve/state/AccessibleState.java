package ampcontrol.model.training.model.evolve.state;

/**
 * {@link PersistentState} which may also be accessed.
 * @param <T>
 *
 * @author Christian Skärby
 */
public interface AccessibleState<T> extends PersistentState<T> {

    /**
     * Return the current state
     * @return the current state
     */
    T get();

    @Override
    AccessibleState<T> clone();

}
