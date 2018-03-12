package ampControl.admin.service.control;

/**
 * Maps messages to actions in the form of {@link Runnable Runnables}.
 * Actions shall be performed every time the mapped message is received
 *
 * @author Christian Skärby
 */
public interface MessageToActionMap {

    /**
     * Maps the given message to the given action
     *
     * @param message the message to map the action
     * @param action The action to perform when the given message is received
     */
    void mapMessage(String message, Runnable action);

    /**
     * Sets action to perform in case connection to message server (typically the MQTT broker) is lost
     *
     * @param action Action to perform in case connection to message server is lost
     */
    void setConnectionFailedAction(Runnable action);
}
