package ampcontrol.model.training.model.evolve.mutate.layer.blockfunctions;

import ampcontrol.model.training.model.layerblocks.Dense;
import ampcontrol.model.training.model.layerblocks.IdBlock;
import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.LayerSpyAdapter;
import ampcontrol.model.training.model.layerblocks.graph.MockGraphAdapter;
import org.apache.commons.lang.mutable.MutableInt;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link ListFunction}
 *
 * @author Christian Skärby
 */
public class ListFunctionTest {

    /**
     * Test that expected functions are applied
     */
    @Test
    public void apply() {
        final LayerBlockConfig block1 = new IdBlock();
        final LayerBlockConfig block2 = new Dense();
        final MutableInt supplier = new MutableInt(0);

        final Function<Long, LayerBlockConfig> blockFunction = ListFunction.builder()
                .function(nOut -> block1)
                .function(nout -> block2)
                .indexSupplier(size -> supplier.intValue())
                .build();

        assertEquals("Unexpected block!", block1, blockFunction.apply(666L));
        assertEquals("Unexpected block!", block1, blockFunction.apply(1L));
        supplier.setValue(1);

        assertEquals("Unexpected block!", block2, blockFunction.apply(666L));
        assertEquals("Unexpected block!", block2, blockFunction.apply(1L));

    }

    /**
     * Test that allConv2D sets correct blocks
     */
    @Test
    public void allConv2D() {

        final MutableInt supplier = new MutableInt(0);
        final Function<Long, LayerBlockConfig> blockFunction = ListFunction.allConv2D(new Random(666))
                .indexSupplier(size -> supplier.intValue())
                .build();

        final List<Layer> layers = new ArrayList<>();
        final LayerSpyAdapter spyAdapter = new LayerSpyAdapter(
                (layerName, layer, layerInputs) -> layers.add(layer),
                new MockGraphAdapter());

        blockFunction.apply(12L).addLayers(spyAdapter, new LayerBlockConfig.SimpleBlockInfo.Builder().setInputs(new String[] {"test"}).build());
        assertEquals("Incorrect number of layers!", 1, layers.size());
        assertTrue("Incorrect layer: " + layers.get(0) +"!", layers.get(0) instanceof ConvolutionLayer);
        layers.clear();

        supplier.setValue(1);
        blockFunction.apply(12L).addLayers(spyAdapter, new LayerBlockConfig.SimpleBlockInfo.Builder().setInputs(new String[] {"test"}).build());
        assertEquals("Incorrect number of layers!", 2, layers.size());
        assertTrue("Incorrect layer: " + layers.get(0) +"!", layers.get(0) instanceof ConvolutionLayer);
        assertTrue("Incorrect layer: " + layers.get(1) +"!", layers.get(1) instanceof BatchNormalization);
        layers.clear();

        supplier.setValue(2);
        blockFunction.apply(12L).addLayers(spyAdapter, new LayerBlockConfig.SimpleBlockInfo.Builder().setInputs(new String[] {"test"}).build());
        assertEquals("Incorrect number of layers!", 2, layers.size());
        assertTrue("Incorrect layer: " + layers.get(0) +"!", layers.get(0) instanceof BatchNormalization);
        assertTrue("Incorrect layer: " + layers.get(1) +"!", layers.get(1) instanceof ConvolutionLayer);
        layers.clear();

        supplier.setValue(3);
        blockFunction.apply(12L).addLayers(spyAdapter, new LayerBlockConfig.SimpleBlockInfo.Builder().setInputs(new String[] {"test"}).build());
        assertEquals("Incorrect number of layers!", 2, layers.size());
        assertTrue("Incorrect layer: " + layers.get(0) +"!", layers.get(0) instanceof ConvolutionLayer);
        assertTrue("Incorrect layer: " + layers.get(1) +"!", layers.get(1) instanceof BatchNormalization);
    }
}