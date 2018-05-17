package ampcontrol.model.training.data;

import java.util.List;
import java.util.function.Supplier;

/**
 * Supplies labels in the order of the provided list.
 * @param <T> Type to supply
 *
 * @author Christian Skärby
 */
public class CyclingLabelSupplier<T> implements Supplier<T> {

    private final List<T> labels;
    private int ind = -1;

    public CyclingLabelSupplier(List<T> labels) {
        this.labels = labels;
    }

    @Override
    public synchronized T get() {
        ind++;
        ind %= labels.size();
        return labels.get(ind);
    }
}
