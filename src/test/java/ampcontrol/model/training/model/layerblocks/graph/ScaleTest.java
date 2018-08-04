package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.ScaleVertex;
import org.junit.Test;

import java.util.function.Function;

/**
 * Test cases for {@link Scale}
 *
 * @author Christian Skärby
 */
public class ScaleTest {

    /**
     * Test addLayers. Tests internals which is bad, but CBA to do anything else for such a simple class.
     */
    @Test
    public void addLayers() {
        final Function<GraphVertex, Boolean> vertextChecker = vertex -> vertex instanceof ScaleVertex;
        final LayerBlockConfig toTest = new Scale(2);
        SimpleVertexProbeAdapter.testSimpleVertexBlock(vertextChecker, toTest);
    }
}