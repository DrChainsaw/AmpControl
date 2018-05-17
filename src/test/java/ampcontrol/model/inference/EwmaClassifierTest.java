package ampcontrol.model.inference;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link EwmaClassifier}.
 *
 * @author Christian Skärby
 */
public class EwmaClassifierTest {

    /**
     * Test that filtering is applied.
     */
    @Test
    public void classify() {
        final double ff = 0.1;
        final INDArray first = Nd4j.create(new double[] {0.2, 0.8});
        final INDArray second = Nd4j.create(new double[] {0.4, 0.6});

        final Classifier classifier = new EwmaClassifier(ff,
                new MockClassifier("mock", 0.666, Arrays.asList(first, second)));
        assertEquals("Incorrect classification!", first, classifier.classify());
        final INDArray expected = Nd4j.create(new double[] {0.22, 0.78});
        assertEquals("Incorrect classificaiton!", expected, classifier.classify());
    }

    /**
     * Test transparency w.r.t accuracy
     */
    @Test
    public void getAccuracy() {
        final MockClassifier mockClassifier = new MockClassifier("mock", 0.666, Nd4j.ones(1));
        final Classifier classifier = new EwmaClassifier(0.333, mockClassifier);
        assertEquals("Incorrect accuracy!", mockClassifier.getAccuracy(), classifier.getAccuracy(), 1e-10);
    }
}