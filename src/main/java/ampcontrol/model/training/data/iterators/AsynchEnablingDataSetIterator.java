package ampcontrol.model.training.data.iterators;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

import java.util.List;

/**
 * {@link MiniEpochDataSetIterator} which uses a {@link ResetableState} to fulfil the {@link MiniEpochDataSetIterator}
 * contract. By doing so, it allows for asynch fetching of data sets which in turn allows for very large data sets at a
 * small cost in performance compared to if the whole data set is stored in working memory.
 *
 * @author Christian Skärby
 */
public class AsynchEnablingDataSetIterator implements MiniEpochDataSetIterator {

    /**
     * Controls the state of the processing pipe line which produces the data.
     */
    public interface ResetableState {

        /**
         * Stores the current state
         */
        void storeCurrentState();

        /**
         * Resets the state to the last saved state
         */
        void restorePreviousState();

    }

    @Override
    public void restartMiniEpoch() {

    }

    @Override
    public int inputColumns() {
        return 0;
    }

    @Override
    public int totalOutcomes() {
        return 0;
    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {

    }

    @Override
    public int batch() {
        return 0;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {

    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return false;
    }

    @Override
    public DataSet next() {
        return null;
    }
}
