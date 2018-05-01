package ampControl.model.training.model.vertex;

import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.Test;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link ChannelMultVertexImpl}
 *
 * @author Christian Skärby
 */
public class ChannelMultVertexImplTest {

    private static final WorkspaceConfiguration wsConf =  WorkspaceConfiguration.builder()
            .initialSize(0)
            .overallocationLimit(0.05)
            .policyLearning(LearningPolicy.FIRST_LOOP)
            .policyReset(ResetPolicy.BLOCK_LEFT)
            .policySpill(SpillPolicy.REALLOCATE)
            .policyAllocation(AllocationPolicy.OVERALLOCATE)
            .build();

    private static final LayerWorkspaceMgr wsMgr = new LayerWorkspaceMgr.Builder()
                    .with(ArrayType.ACTIVATIONS, "WS_ALL_LAYERS_ACT", wsConf)
            .with(ArrayType.ACTIVATION_GRAD, "WS_ACT_GRAD", wsConf)
            .build();
    /**

    /**
     * Test doForward with a small nrof channels and batches
     */
    @Test
    public void doForwardSmallNrofChannels() {
        final int batchSize = 7;
        final int nrofChannels = 5;
        testDoForward(batchSize, nrofChannels);
    }

    /**
     * Test doForward with a large nrof channels and batches
     */
    @Test
    public void doForwardLargeNrofChannels() {
        final int batchSize = 17;
        final int nrofChannels = 25;
        testDoForward(batchSize, nrofChannels);
    }

    /**
     * Test doBackward
     */
    @Test
    public void doBackward() {
        wsMgr.notifyScopeEntered(ArrayType.ACTIVATION_GRAD);
        final int batchSize = 7;
        final int nrofChannels = 5;
        final INDArray convInput = Nd4j.ones(new int[]{batchSize, nrofChannels, 13, 17});
        final INDArray gates = Nd4j.ones(new int[]{batchSize, nrofChannels});
        final INDArray epsilon = convInput.dup();
        final int oneZeroedError = 4;
        epsilon.putScalar(oneZeroedError, 0);
        epsilon.putScalar(11, 0);
        epsilon.putScalar(13, 0);
        final GraphVertex toTest = new ChannelMultVertexImpl(null, "test", 0);
        toTest.setInputs(new INDArray[]{convInput, gates});
        toTest.setEpsilon(epsilon);
        final INDArray[] result = toTest.doBackward(false, wsMgr).getSecond();
        assertEquals("Incorrect conv epsilon!", epsilon, result[0]); // Because convInput is ones
        assertEquals("Incorrect gates mean!",
                epsilon.mean().getDouble(0) / batchSize / nrofChannels * epsilon.length(),
                result[1].mean().getDouble(0), 1e-5); // Wtf precision?
        wsMgr.validateArrayLocation(ArrayType.ACTIVATION_GRAD, epsilon, false, false);
    }

    private static void testDoForward(int batchSize, int nrofChannels) {
        wsMgr.notifyScopeEntered(ArrayType.ACTIVATIONS);
        final INDArray convInput = Nd4j.ones(new int[]{batchSize, nrofChannels, 13, 17});
        final INDArray gates = Nd4j.ones(new int[]{batchSize, nrofChannels});
        final int oneZeroedChannel = 4;
        gates.putScalar(oneZeroedChannel, 0);
        gates.putScalar(11, 0);
        gates.putScalar(13, 0);
        final GraphVertex toTest = new ChannelMultVertexImpl(null, "test", 0);
        toTest.setInputs(new INDArray[]{convInput, gates});
        final INDArray result = toTest.doForward(false, wsMgr);
        assertEquals("Incorrect mean!", gates.mean().getDouble(0), result.mean().getDouble(0), 1e-10);
        assertEquals("Channel was not gated!", 0d,
                result.get(NDArrayIndex.point(0), NDArrayIndex.point(oneZeroedChannel)).mean().getDouble(0), 1e-10);
    }
}