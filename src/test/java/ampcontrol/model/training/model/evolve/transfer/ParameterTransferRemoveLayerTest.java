package ampcontrol.model.training.model.evolve.transfer;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.NoutMutation;
import ampcontrol.model.training.model.evolve.mutate.layer.GraphMutation;
import ampcontrol.model.training.model.evolve.mutate.layer.RemoveVertexFunction;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Optional;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertFalse;

/**
 * Test cases for {@link ParameterTransfer} with {@link GraphMutation} and {@link RemoveVertexFunction}
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
     * Test to remove a convolution layer and mutate Nout of the layer before it
     */
    @Test
    public void removeAndNoutMutateConvBefore() {
        final String toRemove = "toRemove";
        final String toNoutMutate = "toNoutMutate";
        final ComputationGraph graph = GraphUtils.getCnnGraph(toNoutMutate, toRemove, "conv3");
        final ComputationGraph mutatedGraph = removeVertexAndMutateNout(
                graph,
                toRemove,
                toNoutMutate,
                InputType.convolutional(33, 33, 3));
        mutatedGraph.output(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    /**
     * Test to remove a convolution layer and mutate Nout of the layer after it
     */
    @Test
    public void removeAndNoutMutateConvAfter() {
        final String toRemove = "toRemove";
        final String toNoutMutate = "toNoutMutate";
        final ComputationGraph graph = GraphUtils.getCnnGraph("conv1", toRemove, toNoutMutate);
        final ComputationGraph mutatedGraph = removeVertexAndMutateNout(
                graph,
                toRemove,
                toNoutMutate,
                InputType.convolutional(33, 33, 3));
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
        mutatedGraph.output(Nd4j.randn(new long[]{1, 33}));
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

        final ComputationGraphConfiguration.GraphBuilder builder = createBuilder(graph);
        final ComputationGraph newGraph = new ComputationGraph(mutateRemove(builder, layerName)
                .setInputTypes(inputType)
                .build());
        newGraph.init();
        assertEquals("No vertex was removed!", graph.getVertices().length - 1, newGraph.getVertices().length);
        assertFalse("Removed vertex still part of graph!", Optional.ofNullable(newGraph.getVertex(layerName)).isPresent());
        return new ParameterTransfer(graph).transferWeightsTo(newGraph);
    }

    @NotNull
    private ComputationGraphConfiguration.GraphBuilder createBuilder(ComputationGraph graph) {
        return new ComputationGraphConfiguration.GraphBuilder(graph.getConfiguration(),
                new NeuralNetConfiguration.Builder(graph.conf()));
    }

    private final ComputationGraph removeVertexAndMutateNout(
            ComputationGraph graph,
            String layerNameToRemove,
            String layerNameToNoutMutate,
            InputType inputType) {

        final ComputationGraphConfiguration.GraphBuilder builder = createBuilder(graph);
        final ComputationGraph newGraph = new ComputationGraph(
                mutateRemove(
                        mutateNout(builder, layerNameToNoutMutate, graph.layerSize(layerNameToNoutMutate) - 1),
                        layerNameToRemove)
                        .setInputTypes(inputType)
                        .build());
        newGraph.init();
        assertEquals("No vertex was removed!", graph.getVertices().length - 1, newGraph.getVertices().length);
        assertFalse("Removed vertex still part of graph!", Optional.ofNullable(newGraph.getVertex(layerNameToRemove)).isPresent());
        assertFalse("Nout was not changed!",
                graph.layerSize(layerNameToNoutMutate) == newGraph.layerSize(layerNameToNoutMutate));
        return new ParameterTransfer(graph).transferWeightsTo(newGraph);
    }

    @NotNull
    private ComputationGraphConfiguration.GraphBuilder mutateRemove(
            ComputationGraphConfiguration.GraphBuilder builder,
            String layerName) {
        return new GraphMutation(() -> Stream.of(GraphMutation.GraphMutationDescription.builder()
                .mutation(new RemoveVertexFunction(layerName))
                .build()))
                .mutate(builder);
    }

    @NotNull
    private ComputationGraphConfiguration.GraphBuilder mutateNout(
            ComputationGraphConfiguration.GraphBuilder builder,
            String layerName,
            long newNout) {
        return new NoutMutation(
                () -> Stream.of(NoutMutation.NoutMutationDescription.builder()
                        .layerName(layerName)
                        .mutateNout(nOut -> newNout)
                        .build()))
                .mutate(builder);
    }
}
