package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.graph.MockGraphAdapter;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link IdBlock}
 */
public class IdBlockTest {

    /**
     * Test that block is transparent
     */
    @Test
    public void addLayers() {
        final LayerBlockConfig.BlockInfo info = new LayerBlockConfig.SimpleBlockInfo.Builder().build();
        assertEquals("Not identity!", info, new IdBlock().addLayers(new MockGraphAdapter(), info));
    }
}