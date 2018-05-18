package ampcontrol.model.training.model.validation.listen;

import java.util.function.Consumer;

/**
 * Invokes a {@link Consumer} in a new thread.
 *
 * @param <T>
 *
 * @author Christian Skärby
 */
public class NewThread<T> implements Consumer<T> {

    private final Consumer<T> consumer;

    /**
     * Constructor
     * @param consumer the {@link Consumer}
     */
    public NewThread(Consumer<T> consumer) {
        this.consumer = consumer;
    }

    @Override
    public void accept(T t) {
        new Thread(() -> consumer.accept(t)).start();
    }
}
