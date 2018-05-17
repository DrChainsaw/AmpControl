package ampcontrol.admin.service.control.mqtt;

import ampcontrol.admin.service.control.ControlRegistry;
import ampcontrol.admin.service.control.MessageSubscriptionRegistry;
import org.eclipse.paho.client.mqttv3.IMqttDeliveryToken;
import org.eclipse.paho.client.mqttv3.MqttCallback;
import org.eclipse.paho.client.mqttv3.MqttMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

/**
 * {@link MessageSubscriptionRegistry} for MQTT messages.
 *
 * @author Christian Skärby
 */
public class MqttCallbackMap implements MqttCallback, ControlRegistry {

    private static final Logger log = LoggerFactory.getLogger(MqttCallbackMap.class);

    private final Consumer<String> topicSubscriptionListener;
    private final Map<String, Runnable> messageActions = new HashMap<>();
    private final Map<String, Consumer<String>> topicConsumers = new HashMap<>();
    private Runnable connectionFailedAction = () -> log.warn("connection failed!");

    public MqttCallbackMap(Consumer<String> topicSubscriptionListener) {
        this.topicSubscriptionListener = topicSubscriptionListener;
    }

    @Override
    public void registerSubscription(String message, Runnable action) {
        // No problem to support, I just don't like the looks of nestled collections
        if (messageActions.containsKey(message)) {
            throw new IllegalArgumentException("Message " + message + " already mapped to " + messageActions.get(message));
        }
        messageActions.put(message, action);
    }

    public void registerSubscription(String topic, Consumer<String> messageConsumer) {
        // No problem to support, I just don't like the looks of nestled collections
        if (topicConsumers.containsKey(topic)) {
            throw new IllegalArgumentException("Topic " + topic + " already mapped to " + topicConsumers.get(topic));
        }
        topicSubscriptionListener.accept(topic);
        topicConsumers.put(topic, messageConsumer);
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
    public void messageArrived(String topic, MqttMessage mqttMessage) {
        String msg = new String(mqttMessage.getPayload(), StandardCharsets.UTF_8);
        log.info("Got message: " + topic + " msg: " + msg);
        runIfMapped(msg);
        deliverToTopic(topic, msg);
    }

    private void runIfMapped(String msg) {
        Runnable action = messageActions.get(msg);
        if (action != null) {
            log.info("Executing action " + action);
            action.run();
        }
    }

    private void deliverToTopic(String topic, String msg) {
        Consumer<String> messageConsumer = topicConsumers.get(topic);
        if (messageConsumer != null) {
            log.info("Notifying consumer " + messageConsumer);
            messageConsumer.accept(msg);
        }
    }

    @Override
    public void deliveryComplete(IMqttDeliveryToken iMqttDeliveryToken) {
        // No action needed. Class only listens
    }
}
