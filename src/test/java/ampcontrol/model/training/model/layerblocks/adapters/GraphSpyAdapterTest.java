package ampcontrol.model.training.model.layerblocks.adapters;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.ScaleVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * Test cases for {@link GraphSpyAdapter}
 *
 * @author Christian Skärby
 */
public class GraphSpyAdapterTest {

    /**
     * Test that layers can be added and spyed on
     */
    @Test
    public void addLayer() {
        final ProbeAdapter probeAdapter = new ProbeAdapter();
        final ProbeSpy probeSpy = new ProbeSpy();
        new GraphSpyAdapter(probeAdapter, probeSpy).addLayer("addLayer",
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
        new GraphSpyAdapter(probeAdapter, probeSpy).addVertex("addVertex",
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
        new GraphSpyAdapter(probeAdapter, probeSpy).mergeIfMultiple("mergeIf",
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
        new GraphSpyAdapter(probeAdapter, probeSpy).layer(new LayerBlockConfig.SimpleBlockInfo.Builder()
                        .setPrevLayerInd(666)
                        .setInputs(new String[]{"666", "layerInput2"})
                        .build(),
                new DenseLayer.Builder().build());
        assertFalse("Expected input!", probeSpy.inputs.isEmpty());
        assertEquals("Different data in adapter and spy!", probeAdapter.inputs, probeSpy.inputs);
    }

    public static class ProbeAdapter implements GraphBuilderAdapter {

        // yeah yeah yeah. Tests are simple enough...
        public final List<Object> inputs = new ArrayList<>();

        @Override
        public GraphBuilderAdapter addLayer(String layerName, Layer layer, String... layerInputs) {
            inputs.add(layerName);
            inputs.add(layer);
            inputs.add(layerInputs);
            return this;
        }

        @Override
        public GraphBuilderAdapter addVertex(String vertexName, GraphVertex vertex, String... vertexInputs) {
            inputs.add(vertexName);
            inputs.add(vertex);
            inputs.add(vertexInputs);
            return this;
        }

        @Override
        public String mergeIfMultiple(String mergeName, String[] inputs) {
            this.inputs.add(mergeName);
            this.inputs.add(inputs);
            return "";
        }

        @Override
        public LayerBlockConfig.BlockInfo layer(LayerBlockConfig.BlockInfo info, Layer layer) {
            inputs.add(info.getPrevLayerInd() + "");
            inputs.add(layer);
            inputs.add(info.getInputsNames());
            return info;
        }

    }

    public static class ProbeSpy implements GraphSpyAdapter.LayerSpy {
        public final List<Object> inputs = new ArrayList<>();

        @Override
        public void accept(String layerName, Layer layer, String... layerInputs) {
            inputs.add(layerName);
            inputs.add(layer);
            inputs.add(layerInputs);
        }
    }
}