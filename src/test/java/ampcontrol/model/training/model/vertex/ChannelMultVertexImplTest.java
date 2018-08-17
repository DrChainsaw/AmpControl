package ampcontrol.model.training.model.vertex;

import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.Test;
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

    private static final LayerWorkspaceMgr wsMgr = new LayerWorkspaceMgr.Builder().defaultNoWorkspace().build();

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
        final int batchSize = 7;
        final int nrofChannels = 5;
        final INDArray convInput = Nd4j.ones(new int[]{batchSize, nrofChannels, 13, 17});
        final INDArray gates = Nd4j.ones(new int[]{batchSize, nrofChannels});
        final INDArray epsilon = convInput.dup();
        epsilon.putScalar(4, 0d);
        epsilon.putScalar(110, 0d);
        epsilon.putScalar(7300, 0d);
        final GraphVertex toTest = new ChannelMultVertexImpl(null, "test", 0);
        toTest.setInputs(new INDArray[]{convInput, gates});
        toTest.setEpsilon(epsilon);
        final INDArray[] result = toTest.doBackward(false, wsMgr).getSecond();
        assertEquals("Incorrect conv epsilon!", epsilon, result[0]); // Because convInput is ones
        assertEquals("Incorrect gates mean!",
                epsilon.mean(0,1,2,3).mul(epsilon.length()).div(batchSize * nrofChannels).getDouble(0),
                result[1].mean(0,1).getDouble(0), 1e-10);
    }

    private static void testDoForward(int batchSize, int nrofChannels) {
        final INDArray convInput = Nd4j.ones(new int[]{batchSize, nrofChannels, 13, 17});
        final INDArray gates = Nd4j.ones(new int[]{batchSize, nrofChannels});
        final int oneZeroedChannel = 4;
        gates.putScalar(oneZeroedChannel, 0);
        gates.putScalar(11, 0);
        gates.putScalar(13, 0);
        final GraphVertex toTest = new ChannelMultVertexImpl(null, "test", 0);
        toTest.setInputs(new INDArray[]{convInput, gates});
        final INDArray result = toTest.doForward(false, wsMgr);
        assertEquals("Incorrect mean!", gates.mean(1).getDouble(0), result.mean(1).getDouble(0), 1e-10);
        assertEquals("Channel was not gated!", 0d,
                result.get(NDArrayIndex.point(0), NDArrayIndex.point(oneZeroedChannel)).mean(1).getDouble(0), 1e-10);
    }
}