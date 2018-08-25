package ampcontrol.model.training.data.iterators.factory;

import ampcontrol.model.training.data.iterators.DoubleBufferingDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Wraps a {@link DataSetIterator} in a {@link DoubleBufferingDataSetIterator}
 *
 * @param <V> Type of input needed for sourceFactory
 * @author Christian Skärby
 */
public class DoubleBuffered<V> implements DataSetIteratorFactory<DataSetIterator, V> {

    private final int bufferSize;
    private final DataSetIteratorFactory<?, V> sourceFactory;

    /**
     * Constructor
     * @param bufferSize Size of each buffer
     * @param sourceFactory Source factory from which new {@link DataSetIterator}s shall be wrapped.
     */
    public DoubleBuffered(int bufferSize, DataSetIteratorFactory<?, V> sourceFactory) {
        this.bufferSize = bufferSize;
        this.sourceFactory = sourceFactory;
    }

    @Override
    public DataSetIterator create(V input) {
        return new DoubleBufferingDataSetIterator(sourceFactory.create(input), bufferSize);
    }
}
