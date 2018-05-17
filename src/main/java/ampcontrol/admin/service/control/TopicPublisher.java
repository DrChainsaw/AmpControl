package ampcontrol.admin.service.control;

/**
 * Facade interface for {@link org.eclipse.paho.client.mqttv3.IMqttClient#publish}
 *
 * @author Christian Skärby
 */
public interface TopicPublisher {

    /**
     * Publish the given message on the given topic
     *
     * @param topic
     * @param message
     */
    void publish(String topic, String message);
}
