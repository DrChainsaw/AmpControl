package ampcontrol.model.training.data.iterators;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

import java.util.List;

public class MockMiniEpochDataSetIterator implements MiniEpochDataSetIterator {


    @Override
    public void restartMiniEpoch() {
        // Do nothing
    }

    @Override
    public int miniEpochSize() {
        return 1;
    }

    @Override
    public DataSet next(int num) {
        return null;
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
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        // Do nothing
    }

    @Override
    public int batch() {
        return 0;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        // Do nothing
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
        return true;
    }

    @Override
    public DataSet next() {
        return null;
    }
}
