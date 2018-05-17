package ampcontrol.model.training.model.validation.listen;

import java.util.function.Consumer;
import java.util.function.Predicate;

/**
 * Only forwards the received consumable if a certain condition is fulfilled.
 *
 * @param <T>
 *
 * @author Christian Skärby
 */
public class Gated<T> implements Consumer<T> {

    private final Predicate<T> isOpen;
    private final Consumer<T> consumer;

    public Gated(Consumer<T> consumer, Predicate<T> isOpen) {
        this.isOpen = isOpen;
        this.consumer = consumer;
    }

    @Override
    public void accept(T t) {
        if(isOpen.test(t)) {
            consumer.accept(t);
        }
    }
}
