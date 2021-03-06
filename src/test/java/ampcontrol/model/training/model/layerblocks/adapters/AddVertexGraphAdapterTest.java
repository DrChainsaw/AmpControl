package ampcontrol.model.training.model.layerblocks.adapters;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.vertex.EpsilonSpyVertex;
import org.deeplearning4j.nn.conf.graph.ScaleVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;

/**
 * Test cases for {@link AddVertexGraphAdapter}
 *
 * @author Christian Skärby
 */
public class AddVertexGraphAdapterTest {

    /**
     * Test that no spy operation is transparent.
     */
    @Test
    public void addLayerNoSpy() {
        final ProbeAdapter probeAdapter = new ProbeAdapter();
        new AddVertexGraphAdapter(new EpsilonSpyVertex(), layername -> false, probeAdapter).addLayer("addLayer",
                new DenseLayer.Builder().build(),
                "addLayerInput1", "addLayerInput2");
        assertEquals("Adapter did not get input!", 3, probeAdapter.inputs.size());
    }

    /**
     * Test that an exception since it is not possible to intercept
     */
    @Test(expected = IllegalArgumentException.class)
    public void addLayer() {
        final ProbeAdapter probeAdapter = new ProbeAdapter();
        new AddVertexGraphAdapter(new EpsilonSpyVertex(), layername -> true, probeAdapter).addLayer("addLayer",
                new DenseLayer.Builder().build(),
                "addLayerInput1", "addLayerInput2");
    }

    /**
     * Test that no spy operation is transparent.
     */
    @Test
    public void addVertexNoSpy() {
        final ProbeAdapter probeAdapter = new ProbeAdapter();
        new AddVertexGraphAdapter(new EpsilonSpyVertex(), layername -> false, probeAdapter).addVertex("addVertex",
                new ScaleVertex(2),
                "addVertexInput1", "addVertexInput2");
        assertEquals("Adapter did not get input!", 3, probeAdapter.inputs.size());
    }

    /**
     * Test that an exception since it is not possible to intercept
     */
    @Test(expected = IllegalArgumentException.class)
    public void addVertexSpy() {
        final ProbeAdapter probeAdapter = new ProbeAdapter();
        new AddVertexGraphAdapter(new EpsilonSpyVertex(), layername -> true, probeAdapter).addVertex("addVertex",
                new ScaleVertex(2),
                "addVertexInput1", "addVertexInput2"); // exception!
    }

    /**
     * Test that method is forwarded correctly
     */
    @Test
    public void mergeIfMultiple() {
        final ProbeAdapter probeAdapter = new ProbeAdapter();
        new AddVertexGraphAdapter(new EpsilonSpyVertex(), layername->true, probeAdapter).mergeIfMultiple("mergeIf",
                new String[]{"mergeIf1", "mergeIf2"});
        assertEquals("Incorrect data to adapter", 2, probeAdapter.inputs.size());
    }

    @Test
    public void layer() {
        final ProbeAdapter probeAdapter = new ProbeAdapter();
        final String layerName = "666";
        final String spyPrefix = "spy_";
        new AddVertexGraphAdapter(probeAdapter, new EpsilonSpyVertex(), layerName::equals, spyPrefix).layer(new LayerBlockConfig.SimpleBlockInfo.Builder()
                        .setPrevLayerInd(666)
                        .setInputs(new String[]{layerName})
                        .build(),
                new DenseLayer.Builder().build());
        assertEquals("Adapter did not get input!", 6, probeAdapter.inputs.size());
        assertTrue("Expected spy vertex to be present!", probeAdapter.inputs.contains(spyPrefix + layerName));
        assertTrue("Expected spy vertex to be present!", probeAdapter.inputs.contains(new EpsilonSpyVertex())); // Vertexes are value objects...
    }
}