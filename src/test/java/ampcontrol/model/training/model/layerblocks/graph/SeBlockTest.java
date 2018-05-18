package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationSELU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link SeBlock}
 *
 * @author Christian Skärby
 */
public class SeBlockTest {

    /**
     * Test that name is consistent with configuration
     */
    @Test
    public void name() {
        assertEquals("Incorrect name!", "se_gpmm_3p45_SELU", new SeBlock()
                .setGlobPool(new GlobMeanMax())
                .setReduction(3.45)
                .setActivation(new ActivationSELU()).name());
    }

    /**
     * Test that it is possible to configure and run the SE-block. Not easy to verify that the SE block actually works or not...
     */
    @Test
    public void addLayers() {
        final int reduction = 4;
        final int batchSize = 11;
        final int nrofChannels = 7 * reduction;
        final int height = 3;
        final int width = 5;
        final INDArray input = Nd4j.create(batchSize, nrofChannels, height, width);
        final LayerBlockConfig toTest = new SeBlock().setReduction(reduction);
        final int prevNrofOutputs = nrofChannels;
        final InputType inputType = InputType.convolutional(input.size(2), input.size(3), input.size(1));

        final ComputationGraph graph = MockGraphAdapter.createRealComputationGraph(toTest, prevNrofOutputs, inputType);

        graph.output(input);
        final int[] expectedShape = {batchSize, nrofChannels*height*width};
        assertArrayEquals("Incorrect activation shape!", expectedShape, graph.getOutputLayer(0).input().shape());
    }

}