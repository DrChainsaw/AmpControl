package ampControl.admin.service.control.mqtt;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

import ampControl.admin.service.control.MessageToActionMap;
import org.eclipse.paho.client.mqttv3.IMqttDeliveryToken;
import org.eclipse.paho.client.mqttv3.MqttCallback;
import org.eclipse.paho.client.mqttv3.MqttMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link MessageToActionMap} for MQTT messages.
 *
 * @author Christian Skärby
 */
public class MqttCallbackMap implements MqttCallback, MessageToActionMap {

    private static final Logger log = LoggerFactory.getLogger(MqttCallbackMap.class);

    private final Map<String, Runnable> messageActions = new HashMap<>();
    private Runnable connectionFailedAction = () -> {
        log.warn("connection failed!");
    };

    @Override
    public void mapMessage(String message, Runnable action) {
        // No problem to support, I just don't like the looks of nestled collections
        if (messageActions.containsKey(message)) {
            throw new RuntimeException("Message " + message + " already mapped to " + messageActions.get(message));
        }
        messageActions.put(message, action);
    }

    @Override
    public void setConnectionFailedAction(Runnable action) {
        connectionFailedAction = action;
    }

    @Override
    public void connectionLost(Throwable throwable) {
        log.warn("MQTT connection lost!" + throwable);
        connectionFailedAction.run();
    }

    @Override
    public void messageArrived(String s, MqttMessage mqttMessage) {
        String msg = new String(mqttMessage.getPayload(), StandardCharsets.UTF_8);
        log.info("Got message: " + s + " msg: " + msg);
        runIfMapped(msg);
    }

    private void runIfMapped(String msg) {
        Runnable action = messageActions.get(msg);
        if (action != null) {
            log.info("Executing action " + action);
            action.run();
        }
    }

    @Override
    public void deliveryComplete(IMqttDeliveryToken iMqttDeliveryToken) {
        // No action needed. Class only listens
    }
}
