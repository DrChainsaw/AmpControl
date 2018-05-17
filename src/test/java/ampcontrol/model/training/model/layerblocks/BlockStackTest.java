package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import ampcontrol.model.training.model.layerblocks.graph.MockBlock;
import ampcontrol.model.training.model.layerblocks.graph.MockGraphAdapter;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link BlockStack}
 *
 * @author Christian Skärby
 */
public class BlockStackTest {

    /**
     * Test name
     */
    @Test
    public void name() {
        final int nrofStacks = 17;
        final LayerBlockConfig lbc = new MockBlock(){
            @Override
            public String name() {
                return "afehjy";
            }
        };
        final String result = new BlockStack().setBlockToStack(lbc).setNrofStacks(nrofStacks).name();
        final String expected = nrofStacks + "x_" + lbc.name();
        assertEquals("Incorrect name!", expected, result);
    }

    /**
     * Test addLayers with a {@link BuilderAdapter}
     */
    @Test
    public void addLayersBuilderAdapter() {
        final int nrofStacks = 7;
        final int nrofLayersPerBlock = 5;
        final BuilderAdapter adapter = new MockGraphAdapter();
        final LayerBlockConfig lbc = new MockBlock(nrofLayersPerBlock);
        final LayerBlockConfig.BlockInfo input = new LayerBlockConfig.SimpleBlockInfo.Builder().setPrevLayerInd(666).build();
        final LayerBlockConfig.BlockInfo output = new BlockStack().setBlockToStack(lbc).setNrofStacks(nrofStacks)
                .addLayers(adapter, input);
        final int expectedLayerInd = input.getPrevLayerInd() + nrofLayersPerBlock * nrofStacks;
        assertEquals("Incorrect output!", expectedLayerInd, output.getPrevLayerInd());
    }

    /**
     * Test addLayers with a {@link GraphBuilderAdapter}
     */
    @Test
    public void addLayersGraphAdapter() {
        final int nrofStacks = 7;
        final int nrofLayersPerBlock = 5;
        final GraphBuilderAdapter adapter = new MockGraphAdapter();
        final LayerBlockConfig lbc = new MockBlock(nrofLayersPerBlock);
        final LayerBlockConfig.BlockInfo input = new LayerBlockConfig.SimpleBlockInfo.Builder().setPrevLayerInd(666).build();
        final LayerBlockConfig.BlockInfo output = new BlockStack().setBlockToStack(lbc).setNrofStacks(nrofStacks)
                .addLayers(adapter, input);
        final int expectedLayerInd = input.getPrevLayerInd() + nrofLayersPerBlock * nrofStacks;
        assertEquals("Incorrect output!", expectedLayerInd, output.getPrevLayerInd());
    }
}