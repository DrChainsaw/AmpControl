package ampcontrol.model.training.model.layerblocks.adapters;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.graph.ScaleVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * Test cases for {@link LayerSpyAdapter}
 *
 * @author Christian Skärby
 */
public class LayerSpyAdapterTest {

    /**
     * Test that layers can be added and spyed on
     */
    @Test
    public void addLayer() {
        final ProbeAdapter probeAdapter = new ProbeAdapter();
        final ProbeSpy probeSpy = new ProbeSpy();
        new LayerSpyAdapter(probeSpy, probeAdapter).addLayer("addLayer",
                new DenseLayer.Builder().build(),
                "addLayerInput1", "addLayerInput2");
        assertFalse("Expected input!", probeSpy.inputs.isEmpty());
        assertEquals("Different data in adapter and spy!", probeAdapter.inputs, probeSpy.inputs);
    }

    /**
     * Test that addVertex is called
     */
    @Test
    public void addVertex() {
        final ProbeAdapter probeAdapter = new ProbeAdapter();
        final ProbeSpy probeSpy = new ProbeSpy();
        new LayerSpyAdapter(probeSpy, probeAdapter).addVertex("addVertex",
                new ScaleVertex(2),
                "addVertexInput1", "addVertexInput2");
        assertEquals("Incorrect data to adapter", 3, probeAdapter.inputs.size());
        assertEquals("Incorrect data to spy", 0, probeSpy.inputs.size());
    }

    /**
     * Test that mergeIfMultiple is called
     */
    @Test
    public void mergeIfMultiple() {
        final ProbeAdapter probeAdapter = new ProbeAdapter();
        final ProbeSpy probeSpy = new ProbeSpy();
        new LayerSpyAdapter(probeSpy, probeAdapter).mergeIfMultiple("mergeIf",
                new String[]{"mergeIf1", "mergeIf2"});
        assertEquals("Incorrect data to adapter", 2, probeAdapter.inputs.size());
        assertEquals("Incorrect data to spy", 0, probeSpy.inputs.size());
    }

    /**
     * Test that layers can be added and spyed on
     */
    @Test
    public void layer() {
        final ProbeAdapter probeAdapter = new ProbeAdapter();
        final ProbeSpy probeSpy = new ProbeSpy();
        new LayerSpyAdapter(probeSpy, probeAdapter).layer(new LayerBlockConfig.SimpleBlockInfo.Builder()
                        .setPrevLayerInd(666)
                        .setInputs(new String[]{"666", "layerInput2"})
                        .build(),
                new DenseLayer.Builder().build());
        assertFalse("Expected input!", probeSpy.inputs.isEmpty());
        assertEquals("Different data in adapter and spy!", probeAdapter.inputs, probeSpy.inputs);
    }

    public static class ProbeSpy implements LayerSpyAdapter.LayerSpy {
        public final List<Object> inputs = new ArrayList<>();

        @Override
        public void accept(String layerName, Layer layer, String... layerInputs) {
            inputs.add(layerName);
            inputs.add(layer);
            inputs.add(layerInputs);
        }
    }
}