package ampcontrol.model.training.model.evolve.mutate.layer;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.Mutation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jetbrains.annotations.NotNull;
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
        final String conv2ToRemove = "conv2ToRemove";
        final ComputationGraph graph = GraphUtils.getCnnGraph("conv1", conv2ToRemove, "conv3");

        removeVertex(conv2ToRemove, graph, InputType.convolutional(33, 33, 3));
    }

    /**
     * Test removal of the first layer.
     */
    @Test
    public void removeFirstVertex() {
        final String dense1ToRemove = "dense1ToRemove";
        final ComputationGraph graph = GraphUtils.getGraph(dense1ToRemove, "dense2", "dense3");

        removeVertex(dense1ToRemove, graph, InputType.feedForward(33));
    }

    /**
     * Test to remove a residual layer
     */
    @Test
    public void removeResLayer() {
        final String conv2ToRemove = "conv2ToRemove";
        final ComputationGraph graph = GraphUtils.getResNet("conv1", conv2ToRemove, "conv3");

        removeVertex(conv2ToRemove, graph, InputType.convolutional(33, 33, 3));
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


    @NotNull
    private static ComputationGraph removeVertex(String vertexToRemove, ComputationGraph graph, InputType inputType) {
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
}
