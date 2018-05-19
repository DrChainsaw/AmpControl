package ampcontrol.admin.service;

import ampcontrol.admin.service.control.ControlRegistry;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

public class MockControlRegistry implements ControlRegistry {


    private final Map<String, Runnable> messageToAction = new HashMap<>();
    private final Map<String, Consumer<String>> topicToConsumer = new HashMap<>();

    @Override
    public void registerSubscription(String message, Runnable action) {
        messageToAction.put(message, action);
    }

    @Override
    public void setConnectionFailedAction(Runnable action) {
        //Ignore
    }

    @Override
    public void registerSubscription(String topic, Consumer<String> messageConsumer) {
        topicToConsumer.put(topic, messageConsumer);
    }

    /**
     * Executes the action for the given message
     *
     * @param msg the message
     */
    public void execute(String msg) {
        messageToAction.getOrDefault(msg, () -> {}).run();
    }

    /**
     * Delivers the given message to the listener of the given topic
     * @param topic The topic to deliver the message to
     * @param message The message to deliver
     */
    public void deliver(String topic, String message) {
        topicToConsumer.getOrDefault(topic, str -> {}).accept(message);
    }

    /**
     * Returns true if given topic is registered
     *
     * @param topic the topic
     * @return true if given topic is registered
     */
    public boolean isRegistered(String topic) {
        return topicToConsumer.containsKey(topic);
    }
}
