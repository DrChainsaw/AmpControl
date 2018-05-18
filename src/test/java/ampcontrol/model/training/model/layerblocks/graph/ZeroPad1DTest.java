package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.vertex.ZeroPadding1DVertex;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.junit.Test;

import java.util.function.Function;

/**
 * Test cases for {@link ZeroPad1D}
 *
 * @author Christian Skärby
 */
public class ZeroPad1DTest {

    /**
     * Test addLayers. Tests internals which is bad, but CBA to do anything else for such a simple class.
     */
    @Test
    public void addLayers() {
        final Function<GraphVertex, Boolean> vertextChecker = vertex -> vertex instanceof ZeroPadding1DVertex;
        final LayerBlockConfig toTest = new ZeroPad1D();
        SimpleVertexProbeAdapter.testSimpleVertexBlock(vertextChecker, toTest);
    }
}