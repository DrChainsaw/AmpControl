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
        removeVertex(toRemove, graph, InputType.convolutional(33, 33, 3));

    }

    /**
     * Test to remove a convolution layer and mutate Nout of the layer before it
     */
    @Test
    public void removeAndNoutMutateConvBefore() {
        final String toRemove = "toRemove";
        final String toNoutMutate = "toNoutMutate";
        final ComputationGraph graph = GraphUtils.getCnnGraph(toNoutMutate, toRemove, "conv3");
        removeVertexAndMutateNout(
                graph,
                toRemove,
                toNoutMutate,
                InputType.convolutional(33, 33, 3));

    }

    /**
     * Test to remove a convolution layer and mutate Nout of the layer after it
     */
    @Test
    public void removeAndNoutMutateConvAfter() {
        final String toRemove = "toRemove";
        final String toNoutMutate = "toNoutMutate";
        final ComputationGraph graph = GraphUtils.getCnnGraph("conv1", toRemove, toNoutMutate);
        removeVertexAndMutateNout(
                graph,
                toRemove,
                toNoutMutate,
                InputType.convolutional(33, 33, 3));

    }

    /**
     * Test to remove a dense layer
     */
    @Test
    public void removeDense() {
        final String toRemove = "toRemove";
        final ComputationGraph graph = GraphUtils.getGraph("dense1", toRemove, "dense3");
        removeVertex(toRemove, graph, InputType.feedForward(33));
    }

    /**
     * Test to remove a dense layer which is directly after a conv layer
     */
    @Test
    public void removeDenseAfterConv() {
        final String toRemove = "toRemove";
        final ComputationGraph graph = GraphUtils.getConvToDenseGraph("conv1", toRemove, "dense2");
        removeVertex(toRemove, graph, InputType.convolutional(9, 9, 3));
    }

    /**
     * Test to remove a layer which is just before the output layer
     */
    @Test
    public void removeBeforeOut() {
        final String toRemove = "toRemove";
        final ComputationGraph graph = GraphUtils.getGraphNearOut("dense1", toRemove, "output");
        removeVertex(toRemove, graph, InputType.feedForward(33));
    }

    /**
     * Test to remove a residual layer
     */
    @Test
    public void removeResLayer() {
        final String toRemove = "conv2ToRemove";
        final ComputationGraph graph = GraphUtils.getResNet("conv1", toRemove, "conv3");
        removeVertex(toRemove, graph, InputType.convolutional(33, 33, 3));

    }

    /**
     * Test to remove one out of three convolution layers in a fork.
     */
    @Test
    public void removeForkPath() {
        final String fork2ToRemove = "f2ToRemove";
        final ComputationGraph graph = GraphUtils.getForkNet("beforeFork", "afterFork", "f1", fork2ToRemove, "f3");
        removeVertex(fork2ToRemove, graph, InputType.convolutional(33, 33, 3));

    }

    /**
     * Test to remove one out of three convolution layers in a residual fork.
     */
    @Test
    public void removeForkResPath() {
        final String fork2ToRemove = "f2ToRemove";
        final ComputationGraph graph = GraphUtils.getForkResNet("beforeFork", "afterFork", "f1", fork2ToRemove, "f3");
        removeVertex(fork2ToRemove, graph, InputType.convolutional(33, 33, 3));
    }

    /**
     * Test to remove one out of three convolution layers in a complex double residual fork.
     */
    @Test
    public void removeDoubleForkResPath() {
        final String fork2ToRemove = "f2ToRemove";
        final ComputationGraph graph = GraphUtils.getDoubleForkResNet("beforeFork", "afterFork", "f1", fork2ToRemove, "f3", "f4");
        removeVertex(fork2ToRemove, graph, InputType.convolutional(33, 33, 3));
    }

    /**
     * Test to remove one out of three convolution layers in a complex double residual fork. This layer is connected
     * to a single mergevertex which is thus also removed.
     */
    @Test
    public void removeSingleInDoubleForkResPath() {
        final String fork3ToRemove = "f3ToRemove";
        final ComputationGraph graph = GraphUtils.getDoubleForkResNet("beforeFork", "afterFork", "f1", "f2", fork3ToRemove);
        removeVertex(fork3ToRemove, graph, InputType.convolutional(33, 33, 3));
    }

    /**
     * Test to remove the convolution layer just before the first fork in a complex double residual fork.
     */
    @Test
    public void removeBeforeDoubleForkResPath() {
        final String beforeForkToRemove = "beforeForkToRemove";
        final ComputationGraph graph = GraphUtils.getDoubleForkResNet(beforeForkToRemove, "afterFork", "f1", "f2", "f3");
        removeVertex(beforeForkToRemove, graph, InputType.convolutional(33, 33, 3));
    }

    /**
     * Test to remove the convolution layer just after the last fork in a complex double residual fork.
     */
    @Test
    public void removeAfterDoubleForkResPath() {
        final String afterForkToRemove = "afterForkToRemove";
        final ComputationGraph graph = GraphUtils.getDoubleForkResNet("beforeFork", afterForkToRemove, "f1", "f2", "f3");
        removeVertex(afterForkToRemove, graph, InputType.convolutional(33, 33, 3));
    }

    private final ComputationGraph removeVertex(String layerName, ComputationGraph graph, InputType inputType) {

        final ComputationGraphConfiguration.GraphBuilder builder = createBuilder(graph);
        final ComputationGraph newGraph = new ComputationGraph(mutateRemove(builder, layerName)
                .setInputTypes(inputType)
                .build());
        newGraph.init();

        assertFalse("Vertex was not removed!", Stream.of(newGraph.getVertices()).anyMatch(vertex -> vertex.getVertexName().equals(layerName)));
        assertFalse("Removed vertex still part of graph!", Optional.ofNullable(newGraph.getVertex(layerName)).isPresent());
        final ComputationGraph mutatedGraph = new ParameterTransfer(graph).transferWeightsTo(newGraph);
        long[] shape = inputType.getShape(true);
        shape[0] = 1;
        graph.outputSingle(Nd4j.randn(shape));
        newGraph.outputSingle(Nd4j.randn(shape));
        mutatedGraph.outputSingle(Nd4j.randn(shape));
        return mutatedGraph;
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
        final ComputationGraph mutatedGraph = new ParameterTransfer(graph).transferWeightsTo(newGraph);
        long[] shape = inputType.getShape(true);
        shape[0] = 1;
        graph.outputSingle(Nd4j.randn(shape));
        newGraph.outputSingle(Nd4j.randn(shape));
        mutatedGraph.outputSingle(Nd4j.randn(shape));
        return mutatedGraph;
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
