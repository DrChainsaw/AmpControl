package ampcontrol.model.inference;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link EnsembleWeightedSumClassifier}
 *
 * @author Christian Skärby
 */
public class EnsembleWeightedSumClassifierTest {

    /**
     * Test transparency with a single {@link Classifier} with avg normalization
     */
    @Test
    public void classifySingleAvg() {
        final BiFunction<Double, INDArray, INDArray> normalizer = EnsembleWeightedSumClassifier.avgNormalizer;
        testSingleModelEnsemble(normalizer);
    }

    /**
     * Test ensemble of {@link Classifier Classifiers} with avg normalization
     */
    @Test
    public void classifyMultiAvg() {
        final BiFunction<Double, INDArray, INDArray> normalizer = EnsembleWeightedSumClassifier.avgNormalizer;
        testMultiModelEnsemble(normalizer);
    }

    /**
     * Test ensemble of {@link Classifier Classifiers} with softmax normalization
     */
    @Test
    public void classifyMultiSoftmax() {
        final BiFunction<Double, INDArray, INDArray> normalizer = EnsembleWeightedSumClassifier.softMaxNormalizer;
        testMultiModelEnsemble(normalizer);
    }

    void testMultiModelEnsemble(BiFunction<Double, INDArray, INDArray> normalizer) {
        final List<MockClassifier> mockEnsemble = Stream.of(
                new MockClassifier("c1", 0.7, Nd4j.create(new double[]{0.1, 0.2, 0.4, 0.3})),
                new MockClassifier("c2", 0.3, Nd4j.create(new double[]{0.2, 0.3, 0.2, 0.3})),
                new MockClassifier("c3", 0.9, Nd4j.create(new double[]{0.0, 0.1, 0.1, 0.8}))
        )
                .peek(mockClassifier -> mockClassifier.assertCalled(false))
                .collect(Collectors.toList());

        final Classifier classifier = new EnsembleWeightedSumClassifier(mockEnsemble, normalizer);
        assertEquals("Incorrect accuracy!",
                mockEnsemble.stream().mapToDouble(mc -> mc.getAccuracy()).max().getAsDouble(),
                classifier.getAccuracy(), 1e-10);

        final INDArray result = classifier.classify();
        mockEnsemble.forEach(mockClassifier -> mockClassifier.assertCalled(true));
        assertEquals("Incorrect result!", 1d, result.sum(1).getDouble(0), 1e-10);
        assertEquals("Incorrect result!", 3, result.argMax().getInt(0));
    }

    /**
     * Fails with CPU backend
     */
    @Test
    public void testSum() {
        final INDArray test = Nd4j.create(new double[]{1, 2});
        assertEquals("Incorrect result!", Nd4j.create(new double[] {3}), test.sum(1));
    }


    private static void testSingleModelEnsemble(BiFunction<Double, INDArray, INDArray> normalizer) {
        final List<MockClassifier> mockEnsemble = Arrays.asList(
                new MockClassifier("c1", 0.7, Nd4j.create(new double[]{0.1, 0.2, 0.4, 0.3})));
        mockEnsemble.get(0).assertCalled(false);
        final Classifier classifier = new EnsembleWeightedSumClassifier(mockEnsemble, normalizer);
        assertEquals("Incorrect accuracy!", mockEnsemble.get(0).getAccuracy(), classifier.getAccuracy(), 1e-10);
        final INDArray result = classifier.classify();
        mockEnsemble.get(0).assertCalled(true);
        final INDArray expected = mockEnsemble.get(0).classify();
        assertEquals("Incorrect result!", expected, result);
    }
}