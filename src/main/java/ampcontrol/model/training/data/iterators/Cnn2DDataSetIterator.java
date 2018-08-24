package ampcontrol.model.training.data.iterators;

import ampcontrol.model.training.data.DataProvider;
import ampcontrol.model.training.data.DataProvider.TrainingData;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/**
 * "Main" {@link DataSetIterator} for training and offline evaluation. Transforms {@link TrainingData} from an
 * {@link DataProvider} into {@link DataSet DataSets}.
 *
 * @author Christian Skärby
 */
public class Cnn2DDataSetIterator implements DataSetIterator {

    /**
     *
     */
    private static final long serialVersionUID = -7471337372437967674L;
    private final DataProvider dataProvider;
    private final int batchSize;
    private final List<String> labels;

    private DataSetPreProcessor preProcessor;

    private final static class DataAccumulator implements Consumer<TrainingData> {
        private final int batchSize;
        private final List<String> labels;

        private INDArray featureArr;
        private INDArray labelsArr;
        private int batchCnt = 0;

        private DataAccumulator(int batchSize, List<String> labels) {
            this.batchSize = batchSize;
            this.labels = labels;
            labelsArr = Nd4j.zeros(new int[]{batchSize, labels.size()}, 'f');
        }

        @Override
        public synchronized void accept(TrainingData data) {
            List<double[][]> features = data.result().stream()
                    .collect(Collectors.toList());
            for (int featureInd = 0; featureInd < features.size(); featureInd++) {
                double[][] feature = features.get(featureInd);
                if (featureArr == null) {
                    featureArr = Nd4j.createUninitialized(new int[]{batchSize, features.size(), feature.length, feature[0].length}, 'f');
                }

                featureArr.put(
                        new INDArrayIndex[]{
                                NDArrayIndex.point(batchCnt),
                                NDArrayIndex.point(featureInd),
                                NDArrayIndex.all(),
                                NDArrayIndex.all()},
                        Nd4j.create(feature, 'f'));
            }
            final int labelInd = labels.indexOf(data.getLabel());
            labelsArr.putScalar(batchCnt, labelInd, 1.0);

            batchCnt++;
        }

        private DataSet create() {
            return new DataSet(featureArr, labelsArr);
        }

    }

    public Cnn2DDataSetIterator(DataProvider dataProvider, int batchSize, List<String> labels) {
        this.dataProvider = dataProvider;
        this.batchSize = batchSize;
        this.labels = labels;
    }

    @Override
    public boolean hasNext() {
        return true;
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public DataSet next(int num) {

        final DataAccumulator dataAcc = new DataAccumulator(num, labels);
        dataProvider.generateData().limit(num)
                .forEach(dataAcc);
        DataSet ds = dataAcc.create();
        if (preProcessor != null) {
            preProcessor.preProcess(ds);
        }
        return ds;
    }

    @Override
    public int inputColumns() {
        throw new UnsupportedOperationException("Anybody needs this?");
    }

    @Override
    public int totalOutcomes() {
        return labels.size();
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
        throw new UnsupportedOperationException("Can't reset!");
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return labels;
    }

}
