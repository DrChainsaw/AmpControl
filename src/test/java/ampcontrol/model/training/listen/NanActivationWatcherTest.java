package ampcontrol.model.training.listen;

import com.google.common.collect.ImmutableMap;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * Test cases for {@link NanActivationWatcher}
 *
 * @author Christian Skärby
 */
public class NanActivationWatcherTest {

    /**
     * Test onForwardPass without NaNs
     */
    @Test
    public void onForwardPassOk() {
        new NanActivationWatcher().onForwardPass(new MockModel(), Arrays.asList(
                Nd4j.ones(2,3),
                Nd4j.zeros(1,2,3,4)));
    }

    /**
     * Test onForwardPass without NaNs
     */
    @Test
    public void onForwardPassOkCompGraph() {
        new NanActivationWatcher().onForwardPass(new MockModel(), ImmutableMap.<String, INDArray>builder()
                .put("aa", Nd4j.create(1,2,3))
                .put("bb", Nd4j.ones(3,4,5))
        .build());
    }

    /**
     * Test onForwardPass with NaNs
     */
    @Test(expected = RuntimeException.class)
    public void onForwardPassNaN() {
        new NanActivationWatcher().onForwardPass(new MockModel(), Arrays.asList(
                Nd4j.ones(2,3),
                Nd4j.create(new double[] {0, 2, Double.NaN, 666})));
    }

    /**
     * Test onForwardPass with NaNs
     */
    @Test(expected = RuntimeException.class)
    public void onForwardPassNaNCompGraph() {
        new NanActivationWatcher().onForwardPass(new MockModel(), ImmutableMap.<String, INDArray>builder()
                .put("aa", Nd4j.create(1,2,3))
                .put("bb", Nd4j.create(new double[] {0, 2, Double.NaN, 666}))
                .build());
    }
}