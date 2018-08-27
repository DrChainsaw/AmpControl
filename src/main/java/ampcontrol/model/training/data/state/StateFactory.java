package ampcontrol.model.training.data.state;


import java.util.Random;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

/**
 * Creates various stateful objects (e.g. {@link Random}. Main purpose is to be able to collectively store the state
 * and reset it to e.g recreate the exact same data augmentation every time a model is validated.
 *
 * @author Christian Skärby
 */
public interface StateFactory {

    /**
     * Create a new Random.
     * @return a new Random
     */
    Random createNewRandom();

    /**
     * Create a supplier for some state. Changes made in the returned instance are reflected in the given supplier.
     * @param storeFunction Function which defines how to store away the state so that it becomes "unreferenced". E.g.
     *                      {@code a -> new ArrayList<>(a)} if the state is an ArrayList.
     * @param initial Initial value. NOTE: must not be used since changes in the state are not guaranteed to be reflected
     *                in the initial instance.
     * @param <T> Type of state
     * @return a Supplier of the state.
     */
    <T> Supplier<T> createNewStateReference(UnaryOperator<T> storeFunction, T initial);
}
