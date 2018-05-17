package ampcontrol.admin.service.control;

import java.util.function.Consumer;

/**
 * Maintains subscriptions to topics. Registered subscribers will receive message payloads as Strings.
 *
 * @author Christian Skärby
 */
public interface TopicSubscriptionRegistry {

    /**
     * Registers a subscription to a given topic.
     * @param topic
     * @param messageConsumer
     */
    void registerSubscription(String topic, Consumer<String> messageConsumer);
}
