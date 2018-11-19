package ampcontrol.model.training.model.evolve.mutate.layer;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.Mutation;
import ampcontrol.model.training.model.evolve.mutate.util.GraphBuilderUtil;
import ampcontrol.model.training.model.vertex.EpsilonSpyVertex;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
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

    /**
     * Test to remove a layer inside a residual connection just before the elementwise vertex so that the vertices
     * behind it are connected to the the elementwise vertex just before it is removed
     */
    @Test
    public void removeResidualWithGapTest() {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(InputType.convolutional(122, 128, 3))
                .addInputs("input")
                .setOutputs("output")
                .addLayer("fb-1_branch_0_0", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(6).build(), "input")
                .addVertex("spy_fb-1_branch_0_0", new EpsilonSpyVertex(), "fb-1_branch_0_0")
                .addLayer("fb-1_branch_0_1", new BatchNormalization.Builder().build(), "spy_fb-1_branch_0_0")
                .addLayer("fb-1_branch_1_0", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(7).build(), "input")
                .addVertex("spy_fb-1_branch_1_0", new EpsilonSpyVertex(), "fb-1_branch_1_0")
                .addLayer("fb-1_branch_1_1", new BatchNormalization.Builder().build(), "spy_fb-1_branch_1_0")
                .addVertex("rbMv", new MergeVertex(), "fb-1_branch_0_1", "fb-1_branch_1_1")
                .addLayer("1", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(13).build(), "rbMv")
                .addVertex("spy_1", new EpsilonSpyVertex(), "1")
                .addLayer("2", new BatchNormalization.Builder().nOut(13).build(), "spy_1")
                .addVertex("rbAdd0", new ElementWiseVertex(ElementWiseVertex.Op.Add), "rbMv", "2")
                .addLayer("3", new GlobalPoolingLayer(), "rbAdd0")
                .addLayer("4", new DenseLayer.Builder().nOut(13).build(), "3")
                .addVertex("spy_4", new EpsilonSpyVertex(), "4")
                .addLayer("5", new DenseLayer.Builder().nOut(13).build(), "spy_4")
                .addVertex("spy_5", new EpsilonSpyVertex(), "5")
                .addLayer("6", new DenseLayer.Builder().nOut(13).build(), "spy_5")
                .addVertex("spy_6", new EpsilonSpyVertex(), "6")
                .addLayer("output", new CenterLossOutputLayer.Builder().nOut(4).build(), "spy_6")
                .build());
        graph.init();

        removeVertex("2", graph, InputType.convolutional(122, 128, 3));
    }

    /**
     * Test to remove a layer inside a residual connection just before the elementwise vertex so that the vertices
     * behind it are connected to the the elementwise vertex just before it is removed
     */
    @Test
    public void removeSpyAndLayerNearEnd() {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(InputType.feedForward(10))
                .addInputs("input")
                .setOutputs("output")
                .addLayer("1", new DenseLayer.Builder().nOut(19).build(), "input")
                .addVertex("spy_1", new EpsilonSpyVertex(), "1")
                .addLayer("2", new DenseLayer.Builder().nOut(7).build(), "spy_1")
                .addVertex("spy_2", new EpsilonSpyVertex(), "2")
                .addLayer("output", new CenterLossOutputLayer.Builder().nOut(4).build(), "spy_2")
                .build());

        graph.init();
        final ComputationGraphConfiguration.GraphBuilder builer = new ComputationGraphConfiguration.GraphBuilder(
                graph.getConfiguration(), new NeuralNetConfiguration.Builder(graph.conf()));

        new RemoveVertexFunction("spy_2").apply(builer);
        new RemoveVertexFunction("2").apply(builer);

        final ComputationGraph newGraph = new ComputationGraph(builer.build());
        newGraph.init();
        newGraph.fit(new DataSet(Nd4j.randn(new long[]{1, 10}), Nd4j.randn(new long[]{1, 4})));
    }

    /**
     * Test to remove a layer after a residual connection so that also the nIn of one of the residual layers
     * is altered
     */
    @Test
    public void removeAfterResBlock() {
        final InputType inputType = InputType.convolutional(10, 10, 2);
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(inputType)
                .addInputs("input")
                .setOutputs("output")
                .addLayer("1", new ConvolutionLayer.Builder().nOut(5).convolutionMode(ConvolutionMode.Same).build(), "input")
                .addLayer("2", new ConvolutionLayer.Builder().nOut(7).convolutionMode(ConvolutionMode.Same).build(), "1")
                .addLayer("3", new ConvolutionLayer.Builder().nOut(5).convolutionMode(ConvolutionMode.Same).build(), "2")
                .addVertex("add1And3", new ElementWiseVertex(ElementWiseVertex.Op.Add), "1", "3")
                .addLayer("4", new ConvolutionLayer.Builder().nOut(6).convolutionMode(ConvolutionMode.Same).build(), "add1And3")
                .addLayer("gp", new GlobalPoolingLayer(), "4")
                .addLayer("output", new CenterLossOutputLayer.Builder().nOut(4).build(), "gp")
                .build());
        graph.init();

        removeVertex("4", graph, inputType);
    }

    /**
     * Test to remove a layer after a residual connection so that also the nIn of both of the residual layers
     * are altered
     */
    @Test
    public void removeAfterResBlockWithFork() {
        final InputType inputType = InputType.convolutional(10, 10, 2);
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(inputType)
                .addInputs("input")
                .setOutputs("output")
                .addLayer("1", new ConvolutionLayer.Builder().nOut(5).convolutionMode(ConvolutionMode.Same).build(), "input")
                .addLayer("2", new ConvolutionLayer.Builder().nOut(3).convolutionMode(ConvolutionMode.Same).build(), "1")
                .addLayer("3", new ConvolutionLayer.Builder().nOut(2).convolutionMode(ConvolutionMode.Same).build(), "1")
                .addVertex("merge2And3", new MergeVertex(), "2", "3")
                .addVertex("add1And3", new ElementWiseVertex(ElementWiseVertex.Op.Add), "1", "merge2And3")
                .addLayer("4", new ConvolutionLayer.Builder().nOut(7).convolutionMode(ConvolutionMode.Same).build(), "add1And3")
                .addLayer("gp", new GlobalPoolingLayer(), "4")
                .addLayer("output", new CenterLossOutputLayer.Builder().nOut(4).build(), "gp")
                .build());
        graph.init();
        removeVertex("4", graph, inputType);
    }

    /**
     * Test to remove a layer after a residual connection so that also the nIn of both of the residual layers
     * are altered
     */
    @Test
    public void removeRightBeforeFork() {
        final InputType inputType = InputType.convolutional(10, 10, 2);
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(inputType)
                .addInputs("input")
                .setOutputs("output")
                .addLayer("1", new ConvolutionLayer.Builder().nOut(5).convolutionMode(ConvolutionMode.Same).build(), "input")
                .addLayer("2", new ConvolutionLayer.Builder().nOut(3).convolutionMode(ConvolutionMode.Same).build(), "1")
                .addLayer("3", new ConvolutionLayer.Builder().nOut(2).convolutionMode(ConvolutionMode.Same).build(), "1")
                .addVertex("merge2And3", new MergeVertex(), "2", "3")
                .addVertex("add1And3", new ElementWiseVertex(ElementWiseVertex.Op.Add), "1", "merge2And3")
                .addLayer("4", new ConvolutionLayer.Builder().nOut(7).convolutionMode(ConvolutionMode.Same).build(), "add1And3")
                .addLayer("gp", new GlobalPoolingLayer(), "4")
                .addLayer("output", new CenterLossOutputLayer.Builder().nOut(4).build(), "gp")
                .build());
        graph.init();
        removeVertex("2", graph, inputType);
    }

    /**
     * Test to remove a layer in a fork in such a way that nIn of the output after the fork is altered
     */
    @Test
    public void removeInForkWithNInLargest() {
        final InputType inputType = InputType.convolutional(10, 10, 2);
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(inputType)
                .addInputs("input")
                .setOutputs("output")
                .addLayer("1", new ConvolutionLayer.Builder().nOut(11).convolutionMode(ConvolutionMode.Same).build(), "input")
                .addLayer("2", new ConvolutionLayer.Builder().nOut(3).convolutionMode(ConvolutionMode.Same).build(), "1")
                .addLayer("3", new ConvolutionLayer.Builder().nOut(2).convolutionMode(ConvolutionMode.Same).build(), "1")
                .addLayer("4", new ConvolutionLayer.Builder().nOut(5).convolutionMode(ConvolutionMode.Same).build(), "1")
                .addVertex("merge2And3And4", new MergeVertex(), "2", "3", "4")
                .addLayer("5", new ConvolutionLayer.Builder().nOut(7).convolutionMode(ConvolutionMode.Same).build(), "merge2And3And4")
                .addLayer("gp", new GlobalPoolingLayer(), "5")
                .addLayer("output", new CenterLossOutputLayer.Builder().nOut(4).build(), "gp")
                .build());
        graph.init();
        final ComputationGraphConfiguration.GraphBuilder builder = GraphBuilderUtil.toBuilder(graph);
        new RemoveVertexFunction("2").apply(builder);

        final ComputationGraph newGraph = new ComputationGraph(builder.build());
        newGraph.init();
        newGraph.output(Nd4j.randn(new long[]{1, 2, 10, 10}));
    }

    /**
     * Test to remove a layer in a fork in such a way that nIn of the output after the fork is altered
     */
    @Test
    public void removeInForkWithNInLargestAfterBatchNorm() {
        final InputType inputType = InputType.convolutional(10, 10, 2);
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(inputType)
                .addInputs("input")
                .setOutputs("output")
                .addLayer("1", new ConvolutionLayer.Builder().nOut(11).convolutionMode(ConvolutionMode.Same).build(), "input")
                .addLayer("2bn", new BatchNormalization.Builder().nOut(11).build(), "1")
                .addLayer("2", new ConvolutionLayer.Builder().nOut(3).convolutionMode(ConvolutionMode.Same).build(), "2bn")
                .addLayer("3", new ConvolutionLayer.Builder().nOut(2).convolutionMode(ConvolutionMode.Same).build(), "1")
                .addLayer("4", new ConvolutionLayer.Builder().nOut(5).convolutionMode(ConvolutionMode.Same).build(), "1")
                .addVertex("merge2And3And4", new MergeVertex(), "2", "3", "4")
                .addLayer("5", new ConvolutionLayer.Builder().nOut(7).convolutionMode(ConvolutionMode.Same).build(), "merge2And3And4")
                .addLayer("gp", new GlobalPoolingLayer(), "5")
                .addLayer("output", new CenterLossOutputLayer.Builder().nOut(4).build(), "gp")
                .build());
        graph.init();
        final ComputationGraphConfiguration.GraphBuilder builder = GraphBuilderUtil.toBuilder(graph);
        new RemoveVertexFunction("2").apply(builder);

        final ComputationGraph newGraph = new ComputationGraph(builder.build());
        newGraph.init();
        newGraph.output(Nd4j.randn(new long[]{1, 2, 10, 10}));
    }

    /**
     * Test to remove a layer after a fork where one of the layers in the fork is a residual layer
     */
    @Test
    public void removeAfterForkWithResInFork() {
        final InputType inputType = InputType.convolutional(10, 10, 2);
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(inputType)
                .addInputs("input")
                .setOutputs("output")
                .addLayer("1", new ConvolutionLayer.Builder().nOut(11).convolutionMode(ConvolutionMode.Same).build(), "input")
                .addLayer("2", new ConvolutionLayer.Builder().nOut(3).convolutionMode(ConvolutionMode.Same).build(), "1")
                .addLayer("3", new ConvolutionLayer.Builder().nOut(11).convolutionMode(ConvolutionMode.Same).build(), "1")
                .addVertex("add1And3", new ElementWiseVertex(ElementWiseVertex.Op.Add), "1", "3")
                .addLayer("4", new ConvolutionLayer.Builder().nOut(5).convolutionMode(ConvolutionMode.Same).build(), "1")
                .addVertex("merge2And3And4", new MergeVertex(), "2", "add1And3", "4")
                .addLayer("5", new ConvolutionLayer.Builder().nOut(7).convolutionMode(ConvolutionMode.Same).build(), "merge2And3And4")
                .addLayer("gp", new GlobalPoolingLayer(), "5")
                .addLayer("output", new CenterLossOutputLayer.Builder().nOut(4).build(), "gp")
                .build());
        graph.init();
        final ComputationGraphConfiguration.GraphBuilder builder = GraphBuilderUtil.toBuilder(graph);
        new RemoveVertexFunction("5").apply(builder);

        final ComputationGraph newGraph = new ComputationGraph(builder.build());
        newGraph.init();
        newGraph.output(Nd4j.randn(new long[]{1, 2, 10, 10}));
    }

    /**
     * Test to remove a layer in a fork in such a way that nIn of the output after the fork is altered
     */
    @Test
    public void removeInForkWithResBeforeFork() {
        final InputType inputType = InputType.convolutional(10, 10, 2);
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(inputType)
                .addInputs("input")
                .setOutputs("output")
                .addLayer("1", new ConvolutionLayer.Builder().nOut(11).convolutionMode(ConvolutionMode.Same).build(), "input")
                .addLayer("2", new ConvolutionLayer.Builder().nOut(11).convolutionMode(ConvolutionMode.Same).build(), "1")
                .addVertex("add1And2", new ElementWiseVertex(ElementWiseVertex.Op.Add), "1", "2")
                .addLayer("3", new ConvolutionLayer.Builder().nOut(2).convolutionMode(ConvolutionMode.Same).build(), "add1And2")
                .addLayer("4", new ConvolutionLayer.Builder().nOut(3).convolutionMode(ConvolutionMode.Same).build(), "add1And2")
                .addLayer("5", new ConvolutionLayer.Builder().nOut(4).convolutionMode(ConvolutionMode.Same).build(), "add1And2")
                .addVertex("merge2And3And4", new MergeVertex(), "3", "4", "5")
                .addLayer("6", new ConvolutionLayer.Builder().nOut(7).convolutionMode(ConvolutionMode.Same).build(), "merge2And3And4")
                .addLayer("gp", new GlobalPoolingLayer(), "6")
                .addLayer("output", new CenterLossOutputLayer.Builder().nOut(4).build(), "gp")
                .build());
        graph.init();
        final ComputationGraphConfiguration.GraphBuilder builder = GraphBuilderUtil.toBuilder(graph);
        new RemoveVertexFunction("5").apply(builder);

        final ComputationGraph newGraph = new ComputationGraph(builder.build());
        newGraph.init();
        newGraph.output(Nd4j.randn(new long[]{1, 2, 10, 10}));
    }

    @NotNull
    private static ComputationGraph removeVertex(String vertexToRemove, ComputationGraph graph, InputType inputType) {
        final Mutation<ComputationGraphConfiguration.GraphBuilder> mutatation = new GraphMutation(() -> Stream.of(
                GraphMutation.GraphMutationDescription.builder()
                        .mutation(new RemoveVertexFunction(vertexToRemove))
                        .build()));
        final ComputationGraph newGraph = new ComputationGraph(mutatation.mutate(
               GraphBuilderUtil.toBuilder(graph).setInputTypes(inputType))
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
