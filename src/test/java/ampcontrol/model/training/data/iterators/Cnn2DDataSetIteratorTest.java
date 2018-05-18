package ampcontrol.model.training.data.iterators;

import ampcontrol.audio.processing.SingletonDoubleInput;
import ampcontrol.model.training.data.DataProvider;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Test cases for {@link Cnn2DDataSetIterator}
 *
 * @author Christian Skärby
 */
public class Cnn2DDataSetIteratorTest {

    /**
     * Test next
     */
    @Test
    public void next() {
        final int batchSize = 32;
        final List<String> labels = Arrays.asList("A", "B", "C", "D");
        final CheckSumSpy dataProviderSpy = new CheckSumSpy(new MockDataProvider(labels));
        final DataSetIterator iter = new Cnn2DDataSetIterator(dataProviderSpy, batchSize, labels);
        DataSet ds = iter.next();
        assertTrue("Incorrect checksum!", dataProviderSpy.checkSum > 0);
        assertEquals("Incorrect checksum!", dataProviderSpy.checkSum, ds.getFeatures().sum(0,1,2,3).getInt(0));
        INDArray labelInds = ds.getLabels().argMax(1);
        for(int i = 0; i < batchSize; i++) {
            assertEquals("Incorrect label at " +i + "!", dataProviderSpy.labels.get(i), labels.get(labelInds.getInt(i)));
        }
    }

    /**
     * Test next with {@link CachingDataSetIterator}.
     */
    @Test
    public void nextWithCachingIter() {
        final int batchSize = 32;
        final List<String> labels = Arrays.asList("A", "B", "C", "D");
        final CheckSumSpy dataProviderSpy = new CheckSumSpy(new MockDataProvider(labels));
        final DataSetIterator iter = new CachingDataSetIterator(new Cnn2DDataSetIterator(dataProviderSpy, batchSize, labels));
        DataSet ds = iter.next();
        assertTrue("Incorrect checksum!", dataProviderSpy.checkSum > 0);
        assertEquals("Incorrect checksum!", dataProviderSpy.checkSum, ds.getFeatures().sum(0,1,2,3).getInt(0));
        INDArray labelInds = ds.getLabels().argMax(1);
        for(int i = 0; i < batchSize; i++) {
            assertEquals("Incorrect label at " +i + "!", dataProviderSpy.labels.get(i), labels.get(labelInds.getInt(i)));
        }
    }

    private static class MockDataProvider implements DataProvider {

        private int cnt = 0;
        private int labelCnt = 0;
        private final List<String> labels;

        private MockDataProvider(List<String> labels) {
            this.labels = labels;
        }

        @Override
        public synchronized Stream<DataProvider.TrainingData> generateData() {
            return Stream.generate( () ->  new double[][] {{cnt++, cnt++}, {cnt++, cnt++}})
                    .map(result ->new TrainingData(labels.get(labelCnt++ % labels.size()), new SingletonDoubleInput(result)));
        }
    }

    private static class CheckSumSpy implements DataProvider {
        private final DataProvider dp;
        private int checkSum = 0;
        private final List<String> labels = new ArrayList<>();

        private CheckSumSpy(DataProvider dp) {
            this.dp = dp;
        }

        @Override
        public synchronized Stream<TrainingData> generateData() {
            return dp.generateData().peek(this::collect);
        }

        private void collect(TrainingData td) {
            checkSum += td.result().stream().flatMap(Stream::of).flatMapToDouble(DoubleStream::of).sum();
            labels.add(td.getLabel());
        }
    }
}