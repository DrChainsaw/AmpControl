package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.L2NormalizeVertex;
import org.junit.Test;

import java.util.function.Function;

/**
 * Test cases for {@link Norm}
 *
 * @author Christian Skärby
 */
public class NormTest {

    /**
     * Test addLayers. Tests internals which is bad, but CBA to do anything else for such a simple class.
     */
    @Test
    public void addLayers() {
        final Function<GraphVertex, Boolean> vertextChecker = vertex -> vertex instanceof L2NormalizeVertex;
        final LayerBlockConfig toTest = new Norm();
        SimpleVertexProbeAdapter.testSimpleVertexBlock(vertextChecker, toTest);
    }
}