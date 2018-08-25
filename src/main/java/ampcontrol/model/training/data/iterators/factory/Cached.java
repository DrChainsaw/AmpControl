package ampcontrol.model.training.data.iterators.factory;

import ampcontrol.model.training.data.iterators.CachingDataSetIterator;
import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.concurrent.locks.ReentrantLock;

/**
 * Wraps {@link DataSetIterator}s from a given factory in a {@link CachingDataSetIterator}.
 *
 * @param <V> Type of input needed for sourceFactory
 * @author Christian Skärby
 */
public class Cached<V> implements DataSetIteratorFactory<MiniEpochDataSetIterator, V> {

    private final int cacheSize;
    private final DataSetIteratorFactory<?, V> sourceFactory;

    /**
     * Constructor
     * @param cacheSize Size of cache
     * @param sourceFactory Source factory from which new {@link DataSetIterator}s shall be wrapped.
     */
    public Cached(int cacheSize, DataSetIteratorFactory<?, V> sourceFactory) {
        this.cacheSize = cacheSize;
        this.sourceFactory = sourceFactory;
    }


    @Override
    public MiniEpochDataSetIterator create(V input) {
        return new CachingDataSetIterator(sourceFactory.create(input), cacheSize).initCache(new ReentrantLock());
    }
}
