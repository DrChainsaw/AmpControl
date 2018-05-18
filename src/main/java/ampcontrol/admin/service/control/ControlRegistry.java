package ampcontrol.admin.service.control;

/**
 * Registry for controlling an application. Adds to {@link SubscriptionRegistry}
 * by allowing a callback in case the connection is failed.
 *
 * @author Christian Skärby
 */
public interface ControlRegistry extends SubscriptionRegistry {

    /**
     * Sets action to perform in case connection to message server (typically the MQTT broker) is lost
     *
     * @param action Action to perform in case connection to message server is lost
     */
    void setConnectionFailedAction(Runnable action);
}
