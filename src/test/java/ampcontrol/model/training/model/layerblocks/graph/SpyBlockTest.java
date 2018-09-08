package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.Dense;
import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.GraphSpyAdapterTest;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * Test cases for {@link SpyBlock}
 *
 * @author Christian Skärby
 */
public class SpyBlockTest {

    /**
     * Test addLayers. Tests internals which is bad, but CBA to do anything else for such a simple class.
     */
    @Test
    public void addLayers() {
        final GraphSpyAdapterTest.ProbeAdapter probeAdapter = new GraphSpyAdapterTest.ProbeAdapter();
        final GraphSpyAdapterTest.ProbeSpy probeSpy = new GraphSpyAdapterTest.ProbeSpy();
        new SpyBlock(new Dense(), probeSpy)
                .addLayers(probeAdapter, new LayerBlockConfig.SimpleBlockInfo.Builder()
                        .setInputs(new String[]{"666", "addLayer2"})
                        .setPrevLayerInd(666)
                        .build());

        assertFalse("Expected input!", probeSpy.inputs.isEmpty());
        assertEquals("Different data in adapter and spy!", probeAdapter.inputs, probeSpy.inputs);
    }
}