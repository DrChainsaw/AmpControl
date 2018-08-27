package ampcontrol.model.training.data.iterators.factory;

import ampcontrol.model.training.data.DataProvider;
import ampcontrol.model.training.data.iterators.Cnn2DDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;

/**
 * Creates a {@link Cnn2DDataSetIterator}-
 *
 * @author Christian Skärby
 */
public class Cnn2D implements DataSetIteratorFactory<DataSetIterator, DataProvider> {

    private final int batchSize;
    private final List<String> labels;

    /**
     * Constructor
     * @param batchSize Batch size to use
     * @param labels Ordered list of label names
     */
    public Cnn2D(int batchSize, List<String> labels) {
        this.batchSize = batchSize;
        this.labels = labels;
    }

    @Override
    public DataSetIterator create(DataProvider dataProvider) {
        return new Cnn2DDataSetIterator(dataProvider, batchSize, labels);
    }
}
