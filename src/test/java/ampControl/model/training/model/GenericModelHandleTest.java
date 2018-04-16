package ampControl.model.training.model;

import ampControl.model.training.data.iterators.CachingDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import static junit.framework.TestCase.assertTrue;

/**
 * Test cases for {@link GenericModelHandle}.
 *
 * @author Christian Skärby
 */
public class GenericModelHandleTest {

    /**
     * Test fitting a classification model for determining if a number is positive or negative using a simple
     * {@link MultiLayerNetwork} wrapped in a {@link GenericModelHandle}
     */
    @Test
    public void fitPositiveOrNegative() {
        final DataSetIterator addIter = new PosNegDataSetIter();
        final CachingDataSetIterator trainIter = new CachingDataSetIterator(addIter, 2);
        final CachingDataSetIterator evalIter = new CachingDataSetIterator(addIter, 10);
        final ModelHandle toTest = new GenericModelHandle(trainIter, evalIter, new MultiLayerModelAdapter(createPosNegNetwork()),
                "addModel", 0);
        for (int i = 0; i < 30; i++) {
            toTest.fit();
            toTest.resetTraining();
        }
        toTest.eval();
        // Not a good test really. Most likely not the class under tests fault if desired accuracy is not reached.
        assertTrue("Did not reach desired accuracy! Was " + toTest.getBestEvalScore() + "!", toTest.getBestEvalScore() > 0.99);
    }

    private static MultiLayerNetwork createPosNegNetwork() {
        final MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(666)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.1))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new OutputLayer.Builder()
                        .lossFunction(new LossBinaryXENT())
                        .nIn(1)
                        .nOut(2)
                        .activation(new ActivationSoftmax())
                        .build())
                .build());
        model.init();
        return model;
    }

    private static class PosNegDataSetIter implements DataSetIterator {

        private final Random rng = new Random(666);
        private final int batchSize = 10;
        private final double[] positive = {1, 0};
        private final double[] negative = {0, 1};


        @Override
        public DataSet next(int num) {
            final double[] terms1 = rng.ints(num, -10, 10).mapToDouble(i -> i).toArray();
            final double[][] evenOrOdd = DoubleStream.of(terms1).mapToInt(d -> (int) d).mapToObj(i -> i > 0 ? positive : negative).collect(Collectors.toList()).toArray(new double[][]{});
            return new DataSet(Nd4j.create(terms1).transpose(), Nd4j.create(evenOrOdd));
        }

        @Override
        public int totalExamples() {
            return 0;
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
            return true;
        }

        @Override
        public void reset() {

        }

        @Override
        public int batch() {
            return batchSize;
        }

        @Override
        public int cursor() {
            return 0;
        }

        @Override
        public int numExamples() {
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
            return Arrays.asList("positive", "negative");
        }

        @Override
        public boolean hasNext() {
            return true;
        }

        @Override
        public DataSet next() {
            return next(batchSize);
        }
    }

}