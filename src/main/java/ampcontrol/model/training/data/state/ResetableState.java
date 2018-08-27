package ampcontrol.model.training.data.state;

/**
 * State which may be stored/restored.
 *
 * @author Christian Skärby
 */
public interface ResetableState {

    /**
     * Stores the current state
     */
    void storeCurrentState();

    /**
     * Resets the state to the last saved state
     */
    void restorePreviousState();

}
