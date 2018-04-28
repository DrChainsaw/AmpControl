package ampControl.admin.service;

import ampControl.admin.service.control.ControlRegistry;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

public class MockControlRegistry implements ControlRegistry {


    private final Map<String, Runnable> messageToAction = new HashMap<>();

    @Override
    public void registerSubscription(String message, Runnable action) {
        messageToAction.put(message,action);
    }

    @Override
    public void setConnectionFailedAction(Runnable action) {

    }

    @Override
    public void registerSubscription(String topic, Consumer<String> messageConsumer) {

    }

    /**
     * Exectutes the action for the given message
     * @param msg
     */
    public void execute(String msg) {
        messageToAction.getOrDefault(msg, () -> {}).run();
    }
}
