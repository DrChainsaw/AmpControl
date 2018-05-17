package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.Act;
import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link ResBlock}
 *
 * @author Christian Skärby
 */
public class ResBlockTest {

    /**
     * Test that output = L(input) + input when L = identity, i.e. output = 2 * input.
     */
    @Test
    public void addLayers() {
        final String inputName = "input";
        final INDArray input = Nd4j.create(new double[] {77});

        final ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder().graphBuilder()
                .addInputs(inputName)
                .setInputTypes(InputType.feedForward(1));
        final LayerBlockConfig.BlockInfo rb= new ResBlock()
                .setBlockConfig(new Act().setActivation(new ActivationIdentity()))
                .addLayers(graphBuilder,
                        new LayerBlockConfig.SimpleBlockInfo.Builder()
                                .setInputs(new String[]{inputName})
                                .setPrevNrofOutputs(1)
                                .setPrevLayerInd(-1).build());
        // Must have output layer. Must have parameters as well.
        final LayerBlockConfig.BlockInfo output = new DummyOutputLayer().addLayers(graphBuilder, rb);
        graphBuilder.setOutputs(output.getInputsNames());
        final ComputationGraph graph = new ComputationGraph(graphBuilder.build());
        graph.init();
        assertEquals("Incorrect output!", input.mul(2), graph.output(input)[0]);
    }
}