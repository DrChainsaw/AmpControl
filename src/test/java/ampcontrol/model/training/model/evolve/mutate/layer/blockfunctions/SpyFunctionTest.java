package ampcontrol.model.training.model.evolve.mutate.layer.blockfunctions;

import ampcontrol.model.training.model.layerblocks.Dense;
import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.LayerSpyAdapter;
import ampcontrol.model.training.model.layerblocks.graph.MockGraphAdapter;
import org.apache.commons.lang.mutable.MutableBoolean;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;

/**
 * Test cases for {@link SpyFunction}
 *
 * @author Christian Skärby
 */
public class SpyFunctionTest {

    /**
     * Test that spy is applied
     */
    @Test
    public void apply() {
        final MutableBoolean wasCalled = new MutableBoolean(false);
        new SpyFunction(
                adapter -> new LayerSpyAdapter(
                        (layerName, layer, layerInputs) -> wasCalled.setValue(true),
                        adapter),
                nOut -> new Dense()).apply(2L).addLayers(
                        new MockGraphAdapter(),
                new LayerBlockConfig.SimpleBlockInfo.Builder().setInputs(new String[] {"dummy"}).build());
        assertTrue("Was not called!", wasCalled.booleanValue());
    }

    /**
     * Test weight init
     */
    @Test
    public void weightInit() {
        final WeightInit expected = WeightInit.LECUN_UNIFORM;
        final List<Layer> spiedLayers = new ArrayList<>();
        SpyFunction.weightInit(nOut -> new Dense(), expected).apply(1L).addLayers(
                new LayerSpyAdapter((layerName, layer, layerInputs) -> spiedLayers.add(layer), new MockGraphAdapter()),
                new LayerBlockConfig.SimpleBlockInfo.Builder().setInputs(new String[] {"dummy"}).build()
        );
        spiedLayers.stream()
                .filter(layer -> layer instanceof BaseLayer)
                .map(layer -> (BaseLayer)layer)
                .forEach(baseLayer -> assertEquals("Incorrect weight init!", expected, baseLayer.getWeightInit()));
    }
}