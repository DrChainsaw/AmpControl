package ampcontrol.model.training.model.evolve.mutate.layer;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.Mutation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jetbrains.annotations.NotNull;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.stream.Stream;

import static org.junit.Assert.assertFalse;

/**
 * Test cases for {@link GraphMutation} with {@link RemoveVertexFunction}
 *
 * @author Christian Skärby
 */
public class RemoveVertexTest {

    /**
     * Test removal of a convolution layer.
     */
    @Test
    public void removeConvVertex() {
        final String conv1 = "conv1";
        final String conv2 = "conv2";
        final String conv3 = "conv3";
        final ComputationGraph graph = GraphUtils.getCnnGraph(conv1, conv2, conv3);

        removeLayer(conv2, graph, InputType.convolutional(33, 33, 3));
    }

    /**
     * Test removal of the first layer.
     */
    @Test
    public void removeFirstVertex() {
        final String dense1 = "dense1";
        final String dense2 = "dense2";
        final String dense3 = "dense3";
        final ComputationGraph graph = GraphUtils.getGraph(dense1, dense2, dense3);

        removeLayer(dense1, graph, InputType.feedForward(33));
    }

    /**
     * Test to remove a residual layer
     */
    @Test
    public void removeResLayer() {
        final String conv1 = "resConv1";
        final String conv2 = "resConv2ToRemove";
        final String conv3 = "resConv3";
        final ComputationGraph graph = GraphUtils.getResNet(conv1, conv2, conv3);

        removeLayer(conv2, graph, InputType.convolutional(33, 33, 3));
    }

    /**
     * Test to remove one out of three convolution layers in a fork.
     */
    @Test @Ignore
    public void removeForkPath() {
        final String beforeFork = "beforeFork";
        final String afterFork = "afterFork";
        final String fork1 = "fork1";
        final String fork2ToRemove = "fork2ToRemove";
        final String fork3 = "fork3";
        final ComputationGraph graph = GraphUtils.getForkNet(beforeFork, afterFork, fork1, fork2ToRemove, fork3);

        removeLayer(fork2ToRemove, graph, InputType.convolutional(33, 33, 3));
    }

    /**
     * Test to remove one out of three convolution layers in a residual fork.
     */
    @Test @Ignore
    public void removeForkResPath() {
        final String beforeFork = "beforeFork";
        final String afterFork = "afterFork";
        final String fork1 = "fork1";
        final String fork2ToRemove = "fork2ToRemove";
        final String fork3 = "fork3";
        final ComputationGraph graph = GraphUtils.getForkResNet(beforeFork, afterFork, fork1, fork2ToRemove, fork3);

        printNout(beforeFork, graph);
        printNout(fork1, graph);
        printNout(fork2ToRemove, graph);
        printNout(fork3, graph);

        removeLayer(fork2ToRemove, graph, InputType.convolutional(33, 33, 3));

    }


    @NotNull
    private static ComputationGraph removeLayer(String vertexToRemove, ComputationGraph graph, InputType inputType) {
        final Mutation<ComputationGraphConfiguration.GraphBuilder> mutatation = new GraphMutation(() -> Stream.of(
                GraphMutation.GraphMutationDescription.builder()
                        .mutation(new RemoveVertexFunction(vertexToRemove))
                        .build()));
        final ComputationGraph newGraph = new ComputationGraph(mutatation.mutate(
                new ComputationGraphConfiguration.GraphBuilder(graph.getConfiguration(), new NeuralNetConfiguration.Builder(graph.conf())))
                .setInputTypes(inputType)
                .build());
        newGraph.init();

        assertFalse("Expected vertex to be removed!", newGraph.getConfiguration().getVertices().containsKey(vertexToRemove));

        long[] shape = inputType.getShape(true);
        shape[0] = 1;
        graph.outputSingle(Nd4j.randn(shape));
        newGraph.outputSingle(Nd4j.randn(shape));

        return newGraph;
    }

    private void printNout(String beforeFork, ComputationGraph graph) {
        System.out.println("graph " + beforeFork + " nOut: " + graph.layerSize(beforeFork));
    }
}
