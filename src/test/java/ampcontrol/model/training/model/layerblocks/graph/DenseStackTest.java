package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
/**
 * Test cases for {@link DenseStack}. Not fun to test due to very stateful interactions with graph...
 *
 * @author Christian Skärby
 */
public class DenseStackTest {

    /**
     * Test that name is correct.
     */
    @Test
    public void name() {
        final int nrofStacks = 7;
        final String innerName = "frehhtjh";
        final String expectedName = nrofStacks + "ds_" + innerName;
        final String actualName = new DenseStack()
                .setNrofStacks(nrofStacks)
                .setBlockToStack(new MockBlock() {
                    @Override
                    public String name() {
                        return innerName;
                    }
                }).name();
        assertEquals("Incorrect name!", expectedName, actualName);
    }

    /**
     * Test that layers are added correctly to a {@link GraphBuilderAdapter}.
     */
    @Test
    public void addLayersGraph() {
        final int nrofStacks = 11;
        final LayerBlockConfig mockBlock = new MockBlock(3);
        final LayerBlockConfig.BlockInfo info = new LayerBlockConfig.SimpleBlockInfo.Builder()
                .setPrevLayerInd(77)
                .setNameMapper(str -> "mock" + str)
                .setInputs(new String[] {"mock77"})
                .build();

        new DenseStack()
                .setNrofStacks(nrofStacks)
                .setBlockToStack(mockBlock)
                .addLayers(new DenseStackProbe(info), info);
    }

    /**
     * Asserts that Dense stack is correctly built
     */
    private static class DenseStackProbe extends MockGraphAdapter {
        private final List<String> nextExpectedVertexInputs;
        private String nextExpectedLayerInput;

        public DenseStackProbe(LayerBlockConfig.BlockInfo inputInfo) {
            nextExpectedVertexInputs = new ArrayList<>(Arrays.asList(inputInfo.getInputsNames()));
            nextExpectedLayerInput = nextExpectedVertexInputs.get(0);
        }

        @Override
        public GraphBuilderAdapter addLayer(String layerName, Layer layer, String... layerInputs) {
            assertEquals("Incorrect number of inputs!", 1 , layerInputs.length);
            assertEquals("Incorrect layer inputs!", nextExpectedLayerInput, layerInputs[0]);
            nextExpectedVertexInputs.add(layerName);
            return this;
        }

        @Override
        public GraphBuilderAdapter addVertex(String vertexName, GraphVertex vertex, String... vertexInputs) {
            assertEquals("Incorrect vertex inputs!", nextExpectedVertexInputs, Arrays.asList(vertexInputs));
            nextExpectedLayerInput = vertexName;
            return this;
        }
    }
}