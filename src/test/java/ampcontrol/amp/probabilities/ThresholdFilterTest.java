package ampcontrol.amp.probabilities;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;

import static org.junit.Assert.assertEquals;

public class ThresholdFilterTest {

    private static final Interpreter<Boolean> mockInterp = arr -> Collections.singletonList(true);
    private static final Interpreter<Boolean> negaInterp = arr -> Collections.emptyList();


    /**
     * Test that arrays with the right element above the threshold pass through the filter
     */
    @Test
    public void applyPass() {
        final int threshInd = 2;
        final double threshVal = 0.7;
        final Interpreter<Boolean> thresh = new ThresholdFilter<>(threshInd, threshVal, mockInterp);
        final INDArray pass = Nd4j.create(new double[] {0.2, 0.1, 0.71, 0.0});
        assertEquals("Expected result!", mockInterp.apply(pass), thresh.apply(pass));
    }

    /**
     * Test that arrays without the right element above the threshold are masked out by the filter
     */
    @Test
    public void applyMask() {
        final int threshInd = 2;
        final double threshVal = 0.7;
        final Interpreter<Boolean> thresh = new ThresholdFilter<>(threshInd, threshVal, mockInterp);
        final INDArray pass = Nd4j.create(new double[] {200, 100, 0.69, 0.71});
        assertEquals("Expected result!", mockInterp.apply(pass), thresh.apply(pass));
    }
}