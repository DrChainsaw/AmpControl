package ampcontrol.admin.service.control;

/**
 * Maps messages to actions in the form of {@link Runnable Runnables}.
 * Actions shall be performed every time the mapped message is received
 *
 * @author Christian Skärby
 */
public interface MessageSubscriptionRegistry {

    /**
     * Maps the given message to the given action
     *
     * @param message the message to map the action
     * @param action The action to perform when the given message is received
     */
    void registerSubscription(String message, Runnable action);
}
