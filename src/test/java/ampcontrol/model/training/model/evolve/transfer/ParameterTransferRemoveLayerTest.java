package ampcontrol.model.training.model.evolve.transfer;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.layer.GraphMutation;
import ampcontrol.model.training.model.evolve.mutate.layer.RemoveLayerFunction;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Optional;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertFalse;

/**
 * Test cases for {@link ParameterTransfer} with {@link GraphMutation} and {@link RemoveLayerFunction}
 *
 * @author Christian Skärby
 */
public class ParameterTransferRemoveLayerTest {

    /**
     * Test to remove a convolution layer
     */
    @Test
    public void removeConv() {
        final String toRemove = "toRemove";
        final ComputationGraph graph = GraphUtils.getCnnGraph("conv1", toRemove, "conv3");
        final ComputationGraph mutatedGraph = removeVertex(graph, toRemove, InputType.convolutional(33, 33, 3));
        mutatedGraph.output(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    /**
     * Test to remove a dense layer
     */
    @Test
    public void removeDense() {
        final String toRemove = "toRemove";
        final ComputationGraph graph = GraphUtils.getGraph("dense1", toRemove, "dense3");
        final ComputationGraph mutatedGraph = removeVertex(graph, toRemove, InputType.feedForward(33));
        mutatedGraph.output(Nd4j.randn(new long[]{1,33}));
    }

    /**
     * Test to remove a dense layer which is directly after a conv layer
     */
    @Test
    public void removeDenseAfterConv() {
        final String toRemove = "toRemove";
        final ComputationGraph graph = GraphUtils.getConvToDenseGraph("conv1", toRemove, "dense2");
        final ComputationGraph mutatedGraph = removeVertex(graph, toRemove, InputType.convolutional(9, 9, 3));
        mutatedGraph.output(Nd4j.randn(new long[]{1, 3, 9, 9}));
    }

    /**
     * Test to remove a layer which is just before the output layer
     */
    @Test
    public void removeBeforeOut() {
        final String toRemove = "toRemove";
        final ComputationGraph graph = GraphUtils.getGraphNearOut("dense1", toRemove, "output");
        final ComputationGraph mutatedGraph = removeVertex(graph, toRemove, InputType.feedForward(33));
        mutatedGraph.output(Nd4j.randn(new long[]{1, 33}));
    }

    private final ComputationGraph removeVertex(ComputationGraph graph, String layerName, InputType inputType) {

        final ComputationGraph newGraph = new ComputationGraph(
                new GraphMutation(() -> Stream.of(GraphMutation.GraphMutationDescription.builder()
                        .mutation(new RemoveLayerFunction(layerName))
                        .build()))
                        .mutate(new ComputationGraphConfiguration.GraphBuilder(graph.getConfiguration().clone(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                        .setInputTypes(inputType)
                        .build());
        newGraph.init();
        assertEquals("No vertex was removed!", graph.getVertices().length-1, newGraph.getVertices().length);
        assertFalse("Removed vertex still part of graph!", Optional.ofNullable(newGraph.getVertex(layerName)).isPresent());
        return newGraph;
    }
}
