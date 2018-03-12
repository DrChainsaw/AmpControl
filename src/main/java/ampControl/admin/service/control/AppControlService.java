package ampControl.admin.service.control;

import org.eclipse.paho.client.mqttv3.MqttException;

/**
 * Interface for controling the application
 *
 * @author Christian Skärby
 */
public interface AppControlService {

    /**
     * Start the control service. Returns a {@link MessageToActionMap} for mapping messages to actions
     *
     * @return {@link MessageToActionMap}
     * @throws MqttException
     */
    MessageToActionMap start() throws MqttException;

    void stop() throws MqttException;
}
