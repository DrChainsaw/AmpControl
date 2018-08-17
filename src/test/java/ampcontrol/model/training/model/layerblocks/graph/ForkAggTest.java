package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

import static junit.framework.TestCase.assertEquals;

public class ForkAggTest {

    /**
     * Test that name is correct.
     */
    @Test
    public void name() {
        final String delim = "_htrgda_";
        final List<String> innerNames = Arrays.asList("test1", "test2", "test3");
        final String expectedName = "fb_" + String.join(delim, innerNames) + "_bf";
        final ForkAgg forkAgg = new ForkAgg(delim);
        innerNames.stream()
                .map(name -> new MockBlock() {
                    @Override
                    public String name() {
                        return name;
                    }
                }).forEach(forkAgg::add);

        Assert.assertEquals("Incorrect name!", expectedName, forkAgg.name());
    }

    /**
     * Test that output = [L0(input); L1(input); ... LN(input)]
     */
    @Test
    public void addLayers() {
        final String inputName = "input";
        final INDArray input = Nd4j.create(new double[] {-1.23, 2.34}).transpose();
        final double scal0 = 10.11;
        final double scal1 = -13.3;

        final ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder().graphBuilder()
                .addInputs(inputName)
                .setInputTypes(InputType.feedForward(1));
        final LayerBlockConfig.BlockInfo fb= new ForkAgg()
                .add(new Scale(scal0))
                .add(new Scale(scal1))
                .addLayers(graphBuilder,
                        new LayerBlockConfig.SimpleBlockInfo.Builder()
                                .setInputs(new String[]{inputName})
                                .setPrevNrofOutputs((int)input.length())
                                .setPrevLayerInd(-1).build());
        // Must have output layer. Must have parameters as well.
        final LayerBlockConfig.BlockInfo output = new DummyOutputLayer().addLayers(graphBuilder, fb);
        graphBuilder.setOutputs(output.getInputsNames());
        final ComputationGraph graph = new ComputationGraph(graphBuilder.build());
        graph.init();
        DummyOutputLayer.setEyeOutput(graph);

        final INDArray expected = Nd4j.hstack(input.mul(scal0), input.mul(scal1));
        assertEquals("Incorrect output!", expected, graph.output(input)[0]);
    }
}