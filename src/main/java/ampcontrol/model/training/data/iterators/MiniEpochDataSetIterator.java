package ampcontrol.model.training.data.iterators;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * A DataSetIterator which uses "mini epochs" (in lack of a better name). It is mainly used for training multiple models
 * in parallel, e.g. 10 minibatches per model before moving on to the next one. This is typically done by "tricking" DL4J
 * that there is no more data so that the call to fit() returns.
 * <br><br>
 * After a call to {@link #restartMiniEpoch()} the iterator will provide the same set of minibatches as it did last time.
 * Calling {@link #reset()} will make it select a new "mini epoch". Note that each "mini epoch" will trigger DL4J to
 * increase the epoch counter by one step.
 *
 * @author Christian Skärby
 */
public interface MiniEpochDataSetIterator extends DataSetIterator {
    /**
     * Resets the cursor for the cache so that the same {@link DataSet DataSets} will be provided again
     */
    void restartMiniEpoch();

    /**
     * Returns the size (number of {@link DataSet DataSets}) of one mini epoch.
     * @return the size of one mini epoch
     */
    int miniEpochSize();

    @Override
    default DataSet next(int num) {
        throw new UnsupportedOperationException("Not supported!");
    }

}
