package ampcontrol.admin.service;

import ampcontrol.admin.service.control.SubscriptionRegistry;

/**
 * Interface for a service
 *
 */
public interface Service {

    /**
     * Stops the service
     */
    void stop();

    /**
     * Register service
     * @param subscriptionRegistry
     */
    void registerTo(SubscriptionRegistry subscriptionRegistry);
}
