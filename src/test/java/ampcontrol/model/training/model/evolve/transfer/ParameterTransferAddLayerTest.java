package ampcontrol.model.training.model.evolve.transfer;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.layer.BlockMutationFunction;
import ampcontrol.model.training.model.evolve.mutate.layer.GraphMutation;
import ampcontrol.model.training.model.layerblocks.Conv2D;
import ampcontrol.model.training.model.layerblocks.Dense;
import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;

/**
 * Test cases for {@link ParameterTransfer} with {@link GraphMutation} and {@link BlockMutationFunction}.
 *
 * @author Christian Skärby
 */
public class ParameterTransferAddLayerTest {

    /**
     * Test to add a convolution layer after another convolution layer
     */
    @Test
    public void addConv() {
        final String addAfter = "addAfter";
        final ComputationGraph graph = GraphUtils.getCnnGraph("conv1", addAfter, "conv3");
        final ComputationGraph mutatedGraph = addVertex(
                graph,
                addAfter,
                InputType.convolutional(33, 33, 3),
                nOut -> new Conv2D().setNrofKernels(nOut.intValue()));
        mutatedGraph.output(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    /**
     * Test to add a dense layer after another dense layer
     */
    @Test
    public void addDense() {
        final String addAfter = "addAfter";
        final ComputationGraph graph = GraphUtils.getGraph("dense1", addAfter, "dense3");
        final ComputationGraph mutatedGraph = addVertex(
                graph,
                addAfter,
                InputType.feedForward(33),
                nOut -> new Dense().setHiddenWidth(nOut.intValue()));
        mutatedGraph.output(Nd4j.randn(new long[]{1,33}));
    }

    /**
     * Test to add a dense layer between a conv and a dense layer
     */
    @Test
    public void addDenseAfterConv() {
        final String addAfter = "addAfter";
        final ComputationGraph graph = GraphUtils.getConvToDenseGraph("conv1", addAfter, "dense2");
        final ComputationGraph mutatedGraph = addVertex(
                graph,
                addAfter,
                InputType.convolutional(9, 9, 3),
                nOut -> new Dense().setHiddenWidth(nOut.intValue()));
        mutatedGraph.output(Nd4j.randn(new long[]{1, 3, 9, 9}));
    }

    /**
     * Test to add a dense layer just before the output layer
     */
    @Test
    public void addBeforeOut() {
        final String addAfter = "addAfter";
        final ComputationGraph graph = GraphUtils.getGraphNearOut("dense1", addAfter, "output");
        final ComputationGraph mutatedGraph = addVertex(
                graph,
                addAfter,
                InputType.feedForward(33),
                nOut -> new Dense().setHiddenWidth(nOut.intValue()));
        mutatedGraph.output(Nd4j.randn(new long[]{1, 33}));
    }

    private final ComputationGraph addVertex(
            ComputationGraph graph,
            String layerName,
            InputType inputType,
            Function<Long, LayerBlockConfig> blockConfigFunction) {

        final ComputationGraph newGraph = new ComputationGraph(
                new GraphMutation(() -> Stream.of(GraphMutation.GraphMutationDescription.builder()
                        .mutation(new BlockMutationFunction(blockConfigFunction, new String[] {layerName}, str -> str + "_mut"))
                        .build()))
                        .mutate(new ComputationGraphConfiguration.GraphBuilder(graph.getConfiguration().clone(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                        .setInputTypes(inputType)
                        .build());
        newGraph.init();
        assertEquals("No vertex was added!", graph.getVertices().length+1, newGraph.getVertices().length);
        assertTrue("Vertex not added!", Optional.ofNullable(newGraph.getVertex("0_mut")).isPresent());
        return newGraph;
    }
}
