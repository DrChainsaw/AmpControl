package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import org.deeplearning4j.nn.conf.graph.GraphVertex;

import java.util.function.Function;

import static org.junit.Assert.*;

/**
 * Generic testing utility for very simple {@link LayerBlockConfig LayerBlockConfigs}.
 *
 * @author Christian Skärby
 */
public class SimpleVertexProbeAdapter extends MockGraphAdapter {
    private String lastVertexName;
    private String[] vertexInputsName;
    private final Function<GraphVertex, Boolean> vertexChecker;

    public SimpleVertexProbeAdapter(Function<GraphVertex, Boolean> vertexChecker) {
        this.vertexChecker = vertexChecker;
    }

    @Override
    public GraphBuilderAdapter addVertex(String vertexName, GraphVertex vertex, String... vertexInputs) {
        lastVertexName = vertexName;
        vertexInputsName = vertexInputs;
        assertTrue("Incorrect vertex type!", vertexChecker.apply(vertex));
        return this;
    }

    private void assertName(String expected) {
        assertEquals("Incorrect name!", expected, lastVertexName);
    }

    private void assertInputs(String[] expected) {
        assertArrayEquals("Incorrect name!", expected, vertexInputsName);
    }

    public static void testSimpleVertexBlock(Function<GraphVertex, Boolean> vertextChecker, LayerBlockConfig toTest) {
        final int prevLayerInd = 666;
        final String baseName = "fegrg";
        final String expectedName = baseName + (prevLayerInd+1);
        final LayerBlockConfig.BlockInfo info = new LayerBlockConfig.SimpleBlockInfo.Builder()
                .setNameMapper(str -> baseName+str)
                .setPrevLayerInd(prevLayerInd)
                .setInputs(new String[] {"asefe", "ggrgr"})
                .build();
        final SimpleVertexProbeAdapter adapter = new SimpleVertexProbeAdapter(vertextChecker);
        final LayerBlockConfig.BlockInfo output = toTest.addLayers(adapter,info);

        adapter.assertName(expectedName);
        adapter.assertInputs(info.getInputsNames());
        assertEquals("Incorrec layerInd!", prevLayerInd+1, output.getPrevLayerInd());
        assertArrayEquals("Incorrect info!", new String[]{expectedName}, output.getInputsNames());
    }
}
