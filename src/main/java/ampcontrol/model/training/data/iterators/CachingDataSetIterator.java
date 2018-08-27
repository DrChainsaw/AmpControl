package ampcontrol.model.training.data.iterators;


import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.locks.Lock;

/**
 * {@link DataSetIterator} which caches output from another {@link DataSetIterator}. Main use case is validation set but
 * also useful for training multiple models in parallell using the same training data.
 *
 * @author Christian Skärby
 */
public class CachingDataSetIterator implements MiniEpochDataSetIterator {
    private static final Logger log = LoggerFactory.getLogger(CachingDataSetIterator.class);

    /**
     *
     */
    private static final long serialVersionUID = 6874213288810185979L;

    private final DataSetIterator sourceIter;
    private final int nrofItersToCache;
    private final DataSetCache cache;
    private DataSetPreProcessor preProcessor;


    /**
     * Constructor
     *
     * @param sourceIter {@link DataSetIterator} for which a cache shall be created
     */
    public CachingDataSetIterator(DataSetIterator sourceIter) {
        this(sourceIter, 1);
    }

    /**
     * Constructor
     *
     * @param sourceIter       {@link DataSetIterator} for which a cache shall be created
     * @param nrofItersToCache Sets how many iterations from sourceIter will be cached.
     */
    public CachingDataSetIterator(DataSetIterator sourceIter, int nrofItersToCache) {
        this.sourceIter = sourceIter;
        this.nrofItersToCache = nrofItersToCache;
        this.cache = new DataSetCache();
    }

    @Override
    public boolean hasNext() {
        return !cache.isLoaded() || cache.hasNext();
    }

    @Override
    public DataSet next() {

        cache.check(sourceIter, nrofItersToCache);
        DataSet ds = cache.next();

        // Handle pre-processing of data. There can only be one type of PreProcessor between this class and the sourceIter
        if (preProcessor != null) {
            if (sourceIter.getPreProcessor() == null) {
                ds = new DataSet(ds.getFeatures(), ds.getLabels(), ds.getFeaturesMaskArray(), ds.getLabelsMaskArray());
                preProcessor.preProcess(ds);
            } else if (sourceIter.getPreProcessor() != preProcessor) {
                throw new IllegalStateException("Different preprocessors for source and cache! Source: "
                        + sourceIter.getPreProcessor() + " cache: " + preProcessor);
            }
        }
        return ds;
    }

    /**
     * Initializes the cache in a separate thread.
     *
     * @param lock Will wait for lock to be available before updating cache
     * @return The instance to allow for this method to e.g. be chained construction.
     */
    public CachingDataSetIterator initCache(final Lock lock) {
        new Thread(() -> tryInitCache(lock)).start();
        return this;
    }

    private void tryInitCache(Lock lock) {
        lock.lock();
        try {
            cache.check(sourceIter, nrofItersToCache);
        } finally {
            lock.unlock();
        }
    }

    @Override
    public int inputColumns() {
        return sourceIter.inputColumns();
    }

    @Override
    public int totalOutcomes() {
        return sourceIter.totalOutcomes();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return sourceIter.asyncSupported();
    }

    @Override
    public void reset() {
        cache.clear();
        if (sourceIter.resetSupported()) {
            sourceIter.reset();
        }
    }

    @Override
    public int batch() {
        return sourceIter.batch();
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return sourceIter.getPreProcessor();
    }

    /**
     * Sets the DataSetPreProcessor to use. In case this happens before the cache has been created the PreProcessor will
     * be set for the sourceIter so that the cache will consist of pre-processed data. Otherwise the {@link DataSet DataSets}
     * in the cache will be pre-processed instead.
     * <br><br>
     * Beware that in case the sourceIter has another PreProcessor than this class an exception will be thrown. Therefore
     * one must only call this method after cache has been created when training different models with different
     * pre-processing.
     *
     * @param preProcessor the preprocessor to use
     */
    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
        if (!cache.isLoaded()) {
            sourceIter.setPreProcessor(preProcessor);
        }
    }

    @Override
    public List<String> getLabels() {
        return sourceIter.getLabels();
    }

    @Override
    public void restartMiniEpoch() {
        cache.resetCursor();
    }

    @Override
    public int miniEpochSize() {
        return nrofItersToCache;
    }

}
