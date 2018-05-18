package ampcontrol.admin.service;

import ampcontrol.admin.service.control.SubscriptionRegistry;

/**
 * As name suggest. Does not perform any service.
 */
public class NoService implements Service {

    @Override
    public void stop() {
        // Ignore
    }

    @Override
    public void registerTo(SubscriptionRegistry subscriptionRegistry) {
        // Ignore
    }
}
