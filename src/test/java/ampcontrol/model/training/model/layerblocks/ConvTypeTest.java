package ampcontrol.model.training.model.layerblocks;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link ConvType}
 *
 * @author Christian Skärby
 */
public class ConvTypeTest {

    /**
     * Test addLayers with a {@link NeuralNetConfiguration.ListBuilder}.
     */
    @Test
    public void addLayersListBuilder() {
        final int[] inputSize = {7, 13, 17};
        final NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder().list();
        new ConvType(inputSize).addLayers(builder, new LayerBlockConfig.SimpleBlockInfo.Builder().setPrevLayerInd(0).build());
        assertEquals("Incorrect inputType!", InputType.convolutional(inputSize[0], inputSize[1], inputSize[2]), builder.getInputType());
    }

    /**
     * Test addLayers with a {@link ComputationGraphConfiguration.GraphBuilder}.
     */
    @Test
    public void addLayersGraphBuilder() {
        final int[] inputSize = {7, 13, 17};
        final ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder().graphBuilder();
        new ConvType(inputSize).addLayers(builder, new LayerBlockConfig.SimpleBlockInfo.Builder().setPrevLayerInd(0).build());
        assertEquals("Incorrect inputType!", InputType.convolutional(inputSize[0], inputSize[1], inputSize[2]), builder.getNetworkInputTypes().get(0));
    }
}