package ampcontrol.model.training.data.state;

import java.util.function.Supplier;
import java.util.function.UnaryOperator;

/**
 * {@link ResetableState} state. Changes in the returned {@link #currentState} is expected to be reflected in
 * this class.
 *
 * @param <T> Type of state
 * @author Christian Skärby
 */
public class ResetableReferenceState<T> implements ResetableState, Supplier<T> {

    private final UnaryOperator<T> storingFunction;
    private T currentState;
    private T savedState;

    /**
     * Constructor
     *
     * @param storingFunction Defines how to store the state. For example: {@code a -> new ArrayList<>(a)} if the state
     *                        is an ArrayList.
     * @param currentState    The current (initial) state;
     */
    public ResetableReferenceState(UnaryOperator<T> storingFunction, T currentState) {
        this.storingFunction = storingFunction;
        this.currentState = storingFunction.apply(currentState);
        storeCurrentState();
    }


    @Override
    public void storeCurrentState() {
        savedState = storingFunction.apply(currentState);
    }

    @Override
    public void restorePreviousState() {
        currentState = storingFunction.apply(savedState);
    }

    @Override
    public T get() {
        return currentState;
    }
}
