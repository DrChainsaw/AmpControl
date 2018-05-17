package ampcontrol.model.inference;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

class MockClassifier implements Classifier {

    private final String name;
    private final double accuracy;
    private final List<INDArray> probabilities;

    private boolean wasCalled = false;
    private int count = 0;

    MockClassifier(String name, double accuracy, INDArray probabilities) {
        this(name, accuracy, Collections.singletonList(probabilities));
    }

    MockClassifier(String name, double accuracy, List<INDArray> probabilities) {
        this.name = name;
        this.accuracy = accuracy;
        this.probabilities = probabilities;
    }

    @Override
    public INDArray classify() {
        wasCalled = true;
        return probabilities.get(count++ % probabilities.size());
    }

    @Override
    public double getAccuracy() {
        return accuracy;
    }

    void assertCalled(boolean expected) {
        assertEquals("Incorrect usage of classifier " + name + "!", expected, wasCalled);
    }
}
