package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.junit.Test;

import java.util.function.Function;

/**
 * Test cases for {@link PreprocVertex}
 *
 * @author Christian Skärby
 */
public class PreprocVertexTest {

    /**
     * Test addLayers. Tests internals which is bad, but CBA to do anything else for such a simple class.
     */
    @Test
    public void addLayers() {
        final Function<GraphVertex, Boolean> vertextChecker = vertex -> vertex instanceof PreprocessorVertex;
        final LayerBlockConfig toTest = new PreprocVertex();
        SimpleVertexProbeAdapter.testSimpleVertexBlock(vertextChecker, toTest);
    }
}