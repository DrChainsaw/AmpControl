package ampcontrol.model.training.data;

import ampcontrol.model.training.data.state.StateFactory;
import org.apache.commons.lang.mutable.MutableInt;

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
    private final Supplier<MutableInt> ind;

    /**
     * Constructor
     * @param labels List of labels to supply
     * @param stateFactory Factory for state
     */
    public CyclingLabelSupplier(List<T> labels, StateFactory stateFactory) {
        this.labels = labels;
        ind = stateFactory.createNewStateReference(mutInt -> new MutableInt(mutInt.intValue()), new MutableInt(-1));
    }

    @Override
    public synchronized T get() {
        int currInd = ind.get().intValue();
        currInd++;
        currInd %= labels.size();
        ind.get().setValue(currInd);
        return labels.get(currInd);
    }
}
