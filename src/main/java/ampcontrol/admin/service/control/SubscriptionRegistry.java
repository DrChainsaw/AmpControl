package ampcontrol.admin.service.control;

/**
 * Union interface. Does not add any functionality, just signals that an entity is both a {@link MessageSubscriptionRegistry}
 * and a {@link TopicSubscriptionRegistry}.
 *
 * @author Christian Skärby
 */
public interface SubscriptionRegistry extends MessageSubscriptionRegistry, TopicSubscriptionRegistry {

}
