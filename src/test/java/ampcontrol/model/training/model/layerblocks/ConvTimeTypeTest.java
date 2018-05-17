package ampcontrol.model.training.model.layerblocks;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link ConvTimeType}
 *
 * @author Christian Skärby
 */
public class ConvTimeTypeTest {

    /**
     * Test addLayers with a {@link NeuralNetConfiguration.ListBuilder}.
     */
    @Test
    public void addLayersListBuilder() {
        final int[] inputSize = {7, 13};
        final NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder().list();
        new ConvTimeType(inputSize).addLayers(builder, new LayerBlockConfig.SimpleBlockInfo.Builder().setPrevLayerInd(0).build());
        assertEquals("Incorrect inputType!", InputType.recurrent(inputSize[0], inputSize[1]), builder.getInputType());
    }

    /**
     * Test addLayers with a {@link ComputationGraphConfiguration.GraphBuilder}.
     */
    @Test
    public void addLayersGraphBuilder() {
        final int[] inputSize = {7, 13};
        final ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder().graphBuilder();
        new ConvTimeType(inputSize).addLayers(builder, new LayerBlockConfig.SimpleBlockInfo.Builder().setPrevLayerInd(0).build());
        assertEquals("Incorrect inputType!", InputType.recurrent(inputSize[0], inputSize[1]), builder.getNetworkInputTypes().get(0));
    }
}