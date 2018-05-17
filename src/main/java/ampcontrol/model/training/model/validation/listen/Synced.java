package ampcontrol.model.training.model.validation.listen;

import java.util.function.Consumer;

/**
 * Synchronizing decorator for a {@link Consumer}. Intended to be used with {@link NewThread}.
 *
 * @param <T>
 *
 * @author Christian Skärby
 */
public class Synced<T> implements Consumer<T> {

    private final Consumer<T> consumer;
    private final Object syncToken;

    public Synced(Object syncToken, Consumer<T> consumer) {
        this.syncToken = syncToken;
        this.consumer = consumer;
    }

    @Override
    public void accept(T t) {
        synchronized (syncToken) {
            consumer.accept(t);
        }
    }
}
