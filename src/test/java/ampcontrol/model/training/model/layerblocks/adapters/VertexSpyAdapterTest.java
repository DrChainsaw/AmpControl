package ampcontrol.model.training.model.layerblocks.adapters;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.ScaleVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;

/**
 * Test cases for {@link VertexSpyAdapter}
 * 
 * @author Christian Skärby
 */
public class VertexSpyAdapterTest {

    /**
     * Test that addLayer is called
     */
    @Test
    public void addLayer() {
        final ProbeAdapter probeAdapter = new ProbeAdapter();
        final ProbeSpy probeSpy = new ProbeSpy();
        new VertexSpyAdapter(probeSpy, probeAdapter).addLayer("addLayer",
                new DenseLayer.Builder().build(),
                "addLayerInput1", "addLayerInput2");
        assertTrue("Expected no input!", probeSpy.inputs.isEmpty());
        assertEquals("Incorrect data to spy", 0, probeSpy.inputs.size());
    }

    /**
     * Test that addVertex is called
     */
    @Test
    public void addVertex() {
        final ProbeAdapter probeAdapter = new ProbeAdapter();
        final ProbeSpy probeSpy = new ProbeSpy();
        new VertexSpyAdapter(probeSpy, probeAdapter).addVertex("addVertex",
                new ScaleVertex(2),
                "addVertexInput1", "addVertexInput2");
        assertEquals("Incorrect data to adapter", 3, probeAdapter.inputs.size());
        assertEquals("Different data in adapter and spy!", probeAdapter.inputs, probeSpy.inputs);
    }

    /**
     * Test that mergeIfMultiple is called
     */
    @Test
    public void mergeIfMultiple() {
        final ProbeAdapter probeAdapter = new ProbeAdapter();
        final ProbeSpy probeSpy = new ProbeSpy();
        new VertexSpyAdapter(probeSpy, probeAdapter).mergeIfMultiple("mergeIf",
                new String[]{"mergeIf1", "mergeIf2"});
        assertEquals("Incorrect data to adapter", 2, probeAdapter.inputs.size());
        assertEquals("Incorrect data to spy", 0, probeSpy.inputs.size());
    }

    /**
     * Test that layer is called
     */
    @Test
    public void layer() {
        final ProbeAdapter probeAdapter = new ProbeAdapter();
        final ProbeSpy probeSpy = new ProbeSpy();
        new VertexSpyAdapter(probeSpy, probeAdapter).layer(new LayerBlockConfig.SimpleBlockInfo.Builder()
                        .setPrevLayerInd(666)
                        .setInputs(new String[]{"666", "layerInput2"})
                        .build(),
                new DenseLayer.Builder().build());
        assertTrue("Expected input!", probeSpy.inputs.isEmpty());
        assertEquals("Different data in adapter and spy!", 3, probeAdapter.inputs.size());
    }

    private static class ProbeSpy implements VertexSpyAdapter.VertexSpy {
        public final List<Object> inputs = new ArrayList<>();

        @Override
        public void accept(String vertexName, GraphVertex vertex, String... vertexInputs) {
            inputs.add(vertexName);
            inputs.add(vertex);
            inputs.add(vertexInputs);
        }
    }
}