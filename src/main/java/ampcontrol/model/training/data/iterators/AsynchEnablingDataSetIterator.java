package ampcontrol.model.training.data.iterators;

import ampcontrol.model.training.data.state.ResetableState;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;

/**
 * {@link MiniEpochDataSetIterator} which uses a {@link ResetableState} to fulfil the {@link MiniEpochDataSetIterator}
 * contract. By doing so, it allows for asynch fetching of data sets which in turn allows for very large data sets at a
 * small cost in performance compared to if the whole data set is stored in working memory.
 *
 * @author Christian Skärby
 */
public class AsynchEnablingDataSetIterator implements MiniEpochDataSetIterator {

    private final DataSetIterator sourceIter;
    private final ResetableState state;
    private final int miniEpochSize;
    private int miniEpochCount = 0;

    /**
     * Constructor
     *
     * @param sourceIter Iterator for which asynch operation is to be enabled.
     * @param state      State which needs to be stored/restored between mini-epochs
     */
    public AsynchEnablingDataSetIterator(
            DataSetIterator sourceIter,
            ResetableState state,
            int miniEpochSize) {
        this.sourceIter = sourceIter;
        this.state = state;
        this.miniEpochSize = miniEpochSize;
    }

    @Override
    public void restartMiniEpoch() {
        miniEpochCount = 0;
        if(sourceIter.resetSupported()) {
            sourceIter.reset();
        }
    }

    @Override
    public int miniEpochSize() {
        return miniEpochSize;
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
        return true;
    }

    @Override
    public void reset() {
        miniEpochCount = 0;
        state.storeCurrentState();
        if (sourceIter.resetSupported()) {
            sourceIter.reset();
        }
    }

    @Override
    public int batch() {
        return sourceIter.batch();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        sourceIter.setPreProcessor(preProcessor);
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return sourceIter.getPreProcessor();
    }

    @Override
    public List<String> getLabels() {
        return sourceIter.getLabels();
    }

    @Override
    public boolean hasNext() {
        return miniEpochCount < miniEpochSize;
    }

    @Override
    public DataSet next() {
        if (!hasNext()) {
            throw new IllegalStateException("Not allowed!");
        }
        if(miniEpochCount == 0) {
            state.restorePreviousState();
        }
        miniEpochCount++;
        return sourceIter.next();
    }
}
