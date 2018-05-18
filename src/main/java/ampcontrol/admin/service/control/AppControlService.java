package ampcontrol.admin.service.control;

import org.eclipse.paho.client.mqttv3.MqttException;

/**
 * Interface for controling the application
 *
 * @author Christian Skärby
 */
public interface AppControlService {

    /**
     * Start the control service. Returns a {@link MessageSubscriptionRegistry} for mapping messages to actions
     *
     * @return {@link MessageSubscriptionRegistry}
     * @throws MqttException
     */
    ControlRegistry start() throws MqttException;

    void stop() throws MqttException;
}
