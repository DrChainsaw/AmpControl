package ampcontrol.model.inference;

import ampcontrol.audio.ClassifierInputProvider;
import ampcontrol.audio.ClassifierInputProviderFactory;
import com.beust.jcommander.JCommander;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Test cases for {@link ClassifierFromParameters}.
 *
 * @author Christian Skärby
 */
public class ClassifierFromParametersTest {

    private static final String classifierNamesPar = "-classifiers ";
    private static final String ewmaPar = "-cff ";

    /**
     * Test that a single {@link Classifier} is created.
     */
    @Test
    public void singleClassifier() {
        final String classNameA = "A";
        final double accuracy = 0.123;
        final INDArray probabilities = Nd4j.create(new double[] {0.1, 0.2, 0.3});
        final Classifier mockClassifier = new MockClassifier(classNameA, accuracy, probabilities);
        final MockClassifierFactory factory = new MockClassifierFactory(Collections.singletonMap(classNameA, mockClassifier));
        final ClassifierFromParameters classifierFromParameters = new ClassifierFromParameters();
        final String argStr = classifierNamesPar + classNameA + " " + ewmaPar + 1;
        JCommander.newBuilder().addObject(classifierFromParameters)
                .build()
                .parse(argStr.split(" "));
        classifierFromParameters.setFactory(factory);

        try {
            final Classifier resultClassifier = classifierFromParameters.getClassifier(createMockInputFactory());
            factory.assertExpected(Collections.singleton(classNameA));
            assertEquals("Incorrect output!", probabilities, resultClassifier.classify());
            assertEquals("Incorect accuracy!", accuracy, resultClassifier.getAccuracy(), 1e-10);
        } catch (IOException e) {
            fail(e.getMessage());
        }
    }

    /**
     * Test that an ensemble {@link Classifier Classifiers} is created.
     */
    @Test
    public void ensembleClassifier() {
        final int nrofClassifiers = 10;
        final Map<String, MockClassifier> mockClassifiers = IntStream.range(0,nrofClassifiers)
                .boxed()
                .collect(Collectors.toMap(
                        i -> String.valueOf(i),
                        i -> new MockClassifier(String.valueOf(i),
                                (double)i / nrofClassifiers, // Just to comply with method contract
                                Nd4j.create(new double[] {(double)i / nrofClassifiers})) // Just to comply with method contract
                ));
        final MockClassifierFactory factory = new MockClassifierFactory(mockClassifiers);
        final ClassifierFromParameters classifierFromParameters = new ClassifierFromParameters();
        final String argStr = classifierNamesPar + mockClassifiers.keySet().stream().collect(Collectors.joining(",")) + " " + ewmaPar + 1;
        JCommander.newBuilder().addObject(classifierFromParameters)
                .build()
                .parse(argStr.split(" "));
        classifierFromParameters.setFactory(factory);
        try {
            final Classifier resultClassifier = classifierFromParameters.getClassifier(createMockInputFactory());
            factory.assertExpected(mockClassifiers.keySet());
            resultClassifier.classify();
            mockClassifiers.values().forEach(mockClassifier -> mockClassifier.assertCalled(true));
        } catch (IOException e) {
            fail(e.getMessage());
        }
    }

    private static ClassifierInputProviderFactory createMockInputFactory() {
        return new ClassifierInputProviderFactory() {
            @Override
            public ClassifierInputProvider createInputProvider(String inputDescriptionString) {
                return null;
            }

            @Override
            public ClassifierInputProvider.UpdateHandle finalizeAndReturnUpdateHandle() {
                return null;
            }
        };
    }

    private static class MockClassifierFactory implements ClassifierFactory {

        private final Map<String, ? extends Classifier> classifiers;
        private final Set<String> actual = new HashSet<>();

        public MockClassifierFactory(Map<String, ? extends Classifier> classifiers) {
            this.classifiers = classifiers;
        }

        @Override
        public Classifier create(String path, ClassifierInputProvider inputProvider) {
            actual.add(path);
            return classifiers.get(path);
        }

        private void assertExpected(Set<String> expected) {
            assertEquals("Incorrect classifiers created!", actual, expected);
        }
    }
}