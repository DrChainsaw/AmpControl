package ampcontrol.model.training.data.iterators.factory;

import ampcontrol.model.training.data.iterators.AsynchEnablingDataSetIterator;
import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import ampcontrol.model.training.data.state.ResetableState;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Wraps {@link DataSetIterator}s from a given factory in a {@link AsynchEnablingDataSetIterator}.
 *
 * @param <V> Type of input needed for sourceFactory
 * @author Christian Skärby
 */
public class Asynch<V> implements DataSetIteratorFactory<MiniEpochDataSetIterator, V> {

    private final int miniEpochSize;
    private final ResetableState resetableState;
    private final DataSetIteratorFactory<?, V> sourceFactory;

    /**
     * Constructor
     * @param miniEpochSize Size of mini epoch
     * @param resetableState Resets data pipeline
     * @param sourceFactory Source factory from which new {@link DataSetIterator}s shall be wrapped.
     */
    public Asynch(int miniEpochSize, ResetableState resetableState, DataSetIteratorFactory<?, V> sourceFactory) {
        this.miniEpochSize = miniEpochSize;
        this.resetableState = resetableState;
        this.sourceFactory = sourceFactory;
    }


    @Override
    public MiniEpochDataSetIterator create(V input) {
        return new AsynchEnablingDataSetIterator(sourceFactory.create(input), resetableState, miniEpochSize);
    }
}
