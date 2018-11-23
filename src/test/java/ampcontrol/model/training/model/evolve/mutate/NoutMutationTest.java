package ampcontrol.model.training.model.evolve.mutate;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.vertex.EpsilonSpyVertex;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.ScaleVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link NoutMutation}
 *
 * @author Christian Skärby
 */
public class NoutMutationTest {

    /**
     * Test mutation function
     */
    @Test
    public void mutate() {
        final String mut1 = "mut1";
        final String mut2 = "mut2";
        final String noMut = "noMut";
        final ComputationGraph graph = GraphUtils.getCnnGraph(mut1, mut2, noMut);

        final NoutMutation noutMutation = new NoutMutation(() -> Stream.of(
                NoutMutation.NoutMutationDescription.builder().layerName(mut1).mutateNout(nOut -> 2 * nOut).build(),
                NoutMutation.NoutMutationDescription.builder().layerName(mut2).mutateNout(nOut -> 2 * nOut).build()));
        final ComputationGraph newGraph = new ComputationGraph(noutMutation.mutate(
                new ComputationGraphConfiguration.GraphBuilder(graph.getConfiguration(), new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        assertEquals("Incorrect nOut!", 2 * graph.layerSize(mut1), newGraph.layerSize(mut1));
        assertEquals("Incorrect nOut!", 2 * graph.layerSize(mut2), newGraph.layerSize(mut2));
        assertEquals("Incorrect nOut!", graph.layerSize(noMut), newGraph.layerSize(noMut));

        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    /**
     * Test to decrease nOut in a residual conv layer followed by batchnorm and another residual conv layer.
     */
    @Test
    public void mutateResNet() {
        final String mut1 = "firstResConvToMutate";
        final String mut2 = "secondResConvToMutateToo";
        final String noMut = "afterMutate";
        final ComputationGraph graph = GraphUtils.getResNet(mut1, mut2, noMut);

        final long newNoutFirst = 5;
        final long newNoutSecond = 9;
        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName(mut1)
                                .mutateNout(nOut -> newNoutFirst)
                                .build(),
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName(mut2)
                                .mutateNout(nOut -> newNoutSecond)
                                .build()
                ))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        assertEquals("Incorrect nOut!", newNoutSecond, newGraph.layerSize(mut1));
        assertEquals("Incorrect nOut!", newNoutSecond, newGraph.layerSize(mut2));
        assertEquals("Incorrect nOut!", newNoutSecond, newGraph.layerSize(noMut));

        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    /**
     * Test to increase nOut in a convolution layer which is part of a fork.
     */
    @Test
    public void mutateForkNet() {
        final String beforeFork = "beforeFork";
        final String afterFork = "afterFork";
        final String fork1 = "fork1";
        final String fork2ToMutate = "fork2ToMutate";
        final String fork3 = "fork3";
        final ComputationGraph graph = GraphUtils.getForkNet(beforeFork, afterFork, fork1, fork2ToMutate, fork3);

        final long newNout = 9;
        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName(fork2ToMutate)
                                .mutateNout(nOut -> newNout)
                                .build()))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        assertEquals("incorrect nOut!", graph.layerSize(fork1), newGraph.layerSize(fork1));
        assertEquals("Incorrect nOut!", newNout, newGraph.layerSize(fork2ToMutate));
        assertEquals("incorrect nOut!", graph.layerSize(fork3), newGraph.layerSize(fork3));

        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    /**
     * Test to decrease nOut in a convolution layer which is part of a residual fork.
     */
    @Test
    public void mutateForkResNet() {
        final String beforeFork = "beforeFork";
        final String afterFork = "afterFork";
        final String fork1 = "fork1";
        final String fork2ToMutate = "fork2ToMutate";
        final String fork3 = "fork3";
        final ComputationGraph graph = GraphUtils.getForkResNet(beforeFork, afterFork, fork1, fork2ToMutate, fork3);

        final long newNout = 3;
        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName(fork2ToMutate)
                                .mutateNout(nOut -> newNout)
                                .build()))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        assertEquals("Incorrect nOut!", graph.layerSize(fork1), newGraph.layerSize(fork1));
        assertEquals("Incorrect nOut!", newNout, newGraph.layerSize(fork2ToMutate));
        assertEquals("Incorrect nOut!", graph.layerSize(fork3), newGraph.layerSize(fork3));

        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    /**
     * Test to decrease nOut to 2 in a convolution layer which is just before a residual fork with 3 paths.
     * Expectation is that nOut is decreased to 3 since this is the minimum possible size without removing any layers
     */
    @Test
    public void mutateAlotBeforeForkNet() {
        final String beforeForkToMutate = "beforeForkToMutate";
        final String afterFork = "afterFork";
        final String fork1 = "fork1";
        final String fork2 = "fork2";
        final String fork3 = "fork3";
        final ComputationGraph graph = GraphUtils.getForkResNet(beforeForkToMutate, afterFork, fork1, fork2, fork3);

        final long newNout = 2; // Note: 3 branches -> Not possible without removing one path
        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName(beforeForkToMutate)
                                .mutateNout(nOut -> newNout)
                                .build()))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        assertEquals("Incorrect nOut!", 3, newGraph.layerSize(beforeForkToMutate));
        assertEquals("incorrect nOut!", 1, newGraph.layerSize(fork1));
        assertEquals("Incorrect nOut!", 1, newGraph.layerSize(fork2));
        assertEquals("incorrect nOut!", 1, newGraph.layerSize(fork3));

        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    /**
     * Test to decrease nOut by 1 in a convolution layer which is just before a residual fork.
     * Expectation is that the last path in the fork is decreased by 1 as well in order to
     * maintain the same sizes into the element wise add vertex.
     */
    @Test
    public void mutateALittleBeforeForkNet() {
        final String beforeForkToMutate = "beforeForkToMutate";
        final String afterFork = "afterFork";
        final String fork1 = "fork1";
        final String fork2 = "fork2";
        final String fork3 = "fork3";
        final ComputationGraph graph = GraphUtils.getForkResNet(beforeForkToMutate, afterFork, fork1, fork2, fork3);

        final long noutDelta = 1;
        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName(beforeForkToMutate)
                                .mutateNout(nOut -> nOut - noutDelta)
                                .build()))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        assertEquals("Incorrect nOut!", graph.layerSize(beforeForkToMutate) - 1, newGraph.layerSize(beforeForkToMutate));
        assertEquals("incorrect nOut!", graph.layerSize(fork1), newGraph.layerSize(fork1));
        assertEquals("Incorrect nOut!", graph.layerSize(fork2), newGraph.layerSize(fork2));
        // Why last path? Because NoutMutation does remainder * layerSize[i] / sumLayerSizes[i:end] in a loop with i = fork1, fork2, fork3
        assertEquals("incorrect nOut!", graph.layerSize(fork3) - 1, newGraph.layerSize(fork3));

        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    /**
     * Test to decrease nOut in a convolution layer which is part of a residual fork.
     */
    @Test
    public void mutateDoubleForkResNet() {
        final String beforeFork = "beforeFork";
        final String afterFork = "afterFork";
        final String fork1 = "fork1";
        final String fork2ToMutate = "fork2ToMutate";
        final String fork3 = "fork3";
        final ComputationGraph graph = GraphUtils.getDoubleForkResNet(beforeFork, afterFork, fork1, fork2ToMutate, fork3);

        final long newNout = 4;
        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName(fork2ToMutate)
                                .mutateNout(nOut -> newNout)
                                .build()))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        assertEquals("Incorrect nOut!", graph.layerSize(fork1), newGraph.layerSize(fork1));
        assertEquals("Incorrect nOut!", newNout, newGraph.layerSize(fork2ToMutate));
        assertEquals("Incorrect nOut!", graph.layerSize(fork3), newGraph.layerSize(fork3));

        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    /**
     * Test to mutate layers inside a residual fork
     */
    @Test
    public void mutateResidualForkTwice() {
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
                .addVertex("rbMvInput0", new MergeVertex(), "fb-1_branch_0_1", "fb-1_branch_1_1")
                .addLayer("1", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(13).build(), "rbMvInput0")
                .addVertex("spy_1", new EpsilonSpyVertex(), "1")
                .addLayer("2", new BatchNormalization.Builder().nOut(13).build(), "spy_1")
                .addVertex("rbAdd0", new ElementWiseVertex(ElementWiseVertex.Op.Add), "rbMvInput0", "2")
                .addLayer("3p", new SubsamplingLayer.Builder().kernelSize(2, 2).stride(2, 2).build(), "rbAdd0")
                .addLayer("3", new GlobalPoolingLayer(), "3p")
                .addLayer("4", new DenseLayer.Builder().nOut(13).build(), "3")
                .addVertex("spy_4", new EpsilonSpyVertex(), "4")
                .addLayer("5", new DenseLayer.Builder().nOut(13).build(), "spy_4")
                .addVertex("spy_5", new EpsilonSpyVertex(), "5")
                .addLayer("output", new CenterLossOutputLayer.Builder().nOut(4).build(), "spy_5")
                .build());
        graph.init();

        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName("fb-1_branch_0_0")
                                .mutateNout(nOut -> nOut - 1)
                                .build(),
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName("fb-1_branch_1_0")
                                .mutateNout(nOut -> nOut - 1)
                                .build()))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 122, 128}));
    }

    /**
     * Test to mutate a layer inside a residual fork which is just behind a layer which propagates size changes.
     */
    @Test
    public void mutateResidualForkBehindNonLayerVertex() {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(InputType.convolutional(122, 128, 3))
                .addInputs("input")
                .setOutputs("output")
                .addLayer("1", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(6 + 7 + 8).build(), "input")
                .addLayer("fb-1_branch_0_0", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(6).build(), "1")
                .addLayer("fb-1_branch_0_1", new BatchNormalization.Builder().build(), "fb-1_branch_0_0")
                .addVertex("scale_fb-1_branch_0_1", new ScaleVertex(1), "fb-1_branch_0_1")
                .addLayer("fb-1_branch_1_0", new BatchNormalization.Builder().build(), "1")
                .addLayer("fb-1_branch_1_1", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(7).build(), "fb-1_branch_1_0")
                .addLayer("fb-1_branch_2_0", new BatchNormalization.Builder().build(), "1")
                .addLayer("fb-1_branch_2_1", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(8).build(), "fb-1_branch_2_0")
                .addVertex("rbMvInput0", new MergeVertex(), "scale_fb-1_branch_0_1", "fb-1_branch_1_1", "fb-1_branch_2_1")
                .addVertex("rbAdd0", new ElementWiseVertex(ElementWiseVertex.Op.Add), "rbMvInput0", "1")
                .addLayer("2", new SubsamplingLayer.Builder().kernelSize(2, 2).stride(2, 2).build(), "rbAdd0")
                .addLayer("3", new GlobalPoolingLayer(), "2")
                .addLayer("4", new DenseLayer.Builder().nOut(13).build(), "3")
                .addLayer("output", new CenterLossOutputLayer.Builder().nOut(4).build(), "4")
                .build());
        graph.init();

        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName("fb-1_branch_0_0")
                                .mutateNout(nOut -> nOut - 1)
                                .build()))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 122, 128}));
    }

    /**
     * Test to mutate a layer just before a size transparent fork
     */
    @Test
    public void mutateBeforeSizeTransparentFork() {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(InputType.convolutional(122, 128, 3))
                .addInputs("input")
                .setOutputs("output")
                .addLayer("1", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(7).build(), "input")
                .addLayer("fb1_branch_0_0", new BatchNormalization.Builder().build(), "1")
                .addVertex("scale_fb1_branch_0_0", new ScaleVertex(1),"fb1_branch_0_0")
                .addLayer("fb1_branch_1_0", new BatchNormalization.Builder().build(), "1")
                 .addVertex("rbMvInput0", new MergeVertex(), "scale_fb1_branch_0_0", "fb1_branch_1_0")
                .addLayer("2", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(7).build(), "rbMvInput0")
                .addLayer("3", new GlobalPoolingLayer(), "2")
                .addLayer("output", new CenterLossOutputLayer.Builder().nOut(4).build(), "3")
                .build());
        graph.init();

        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName("1")
                                .mutateNout(nOut -> nOut - 1)
                                .build()))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 122, 128}));
    }

    /**
     * Test to mutate a layer just before a size transparent fork
     */
    @Test
    public void mutateAfterResBeforeSizeTransparentFork() {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(InputType.convolutional(122, 128, 3))
                .addInputs("input")
                .setOutputs("output")
                .addLayer("1", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(7).build(), "input")
                .addLayer("2", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(7).build(), "1")
                .addVertex("add1And2", new ElementWiseVertex(ElementWiseVertex.Op.Add), "1", "2")
                .addLayer("fb1_branch_0_0", new BatchNormalization.Builder().build(), "add1And2")
                .addLayer("fb1_branch_1_0", new BatchNormalization.Builder().build(), "add1And2")
                .addVertex("rbMvInput0", new MergeVertex(), "fb1_branch_0_0", "fb1_branch_1_0")
                .addLayer("3", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(7).build(), "rbMvInput0")
                .addLayer("4", new GlobalPoolingLayer(), "3")
                .addLayer("output", new CenterLossOutputLayer.Builder().nOut(4).build(), "4")
                .build());
        graph.init();

        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName("2")
                                .mutateNout(nOut -> nOut - 1)
                                .build()))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 122, 128}));
    }

    /**
     * Test to mutate a layer just before a size transparent fork
     */
    @Test
    public void mutateBeforeSizeTransparentForkWithRes() {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(InputType.convolutional(122, 128, 3))
                .addInputs("input")
                .setOutputs("output")
                .addLayer("1", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(7).build(), "input")
                .addLayer("fb1_branch_0_0", new BatchNormalization.Builder().build(), "1")
                .addVertex("scale_fb1_branch_0_0", new ScaleVertex(1),"fb1_branch_0_0")
                .addVertex("addScaleAnd1", new ElementWiseVertex(ElementWiseVertex.Op.Add), "1", "scale_fb1_branch_0_0")
                .addLayer("fb1_branch_1_0", new BatchNormalization.Builder().build(), "1")
                .addVertex("rbMvInput0", new MergeVertex(), "addScaleAnd1", "fb1_branch_1_0")
                .addLayer("2", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(7).build(), "rbMvInput0")
                .addLayer("3", new GlobalPoolingLayer(), "2")
                .addLayer("output", new CenterLossOutputLayer.Builder().nOut(4).build(), "3")
                .build());
        graph.init();

        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName("1")
                                .mutateNout(nOut -> nOut - 1)
                                .build()))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 122, 128}));
    }

    /**
     * Test to mutate a layer just before a size transparent fork
     */
    @Test
    public void mutateBeforeSizeTransparentResForkWithRes() {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(InputType.convolutional(122, 128, 3))
                .addInputs("input")
                .setOutputs("output")
                .addLayer("1", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(14).build(), "input")
                .addLayer("2", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(7).build(), "1")
                .addLayer("fb2_branch_0_0", new BatchNormalization.Builder().build(), "2")
                .addVertex("scale_fb2_branch_0_0", new ScaleVertex(1),"fb2_branch_0_0")
                .addVertex("addScaleAnd2", new ElementWiseVertex(ElementWiseVertex.Op.Add), "2", "scale_fb2_branch_0_0")
                .addLayer("fb2_branch_1_0", new BatchNormalization.Builder().build(), "2")
                .addVertex("rbMvInput0", new MergeVertex(), "addScaleAnd2", "fb2_branch_1_0")
                .addVertex("addMergeAnd1", new ElementWiseVertex(ElementWiseVertex.Op.Add), "rbMvInput0","1")
                .addLayer("3", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(7).build(), "addMergeAnd1")
                .addLayer("4", new GlobalPoolingLayer(), "3")
                .addLayer("output", new CenterLossOutputLayer.Builder().nOut(4).build(), "4")
                .build());
        graph.init();
        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 122, 128}));

        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName("1")
                                .mutateNout(nOut -> nOut - 1)
                                .build()))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 122, 128}));
    }

    /**
     * Test to mutate a layer just before a size transparent fork
     */
    @Test
    public void mutateBeforeSizeTransparentResForkWithInnerFork() {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(InputType.convolutional(122, 128, 3))
                .addInputs("input")
                .setOutputs("output")
                .addLayer("1", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(15).build(), "input")
                .addLayer("2", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(5).build(), "1")
                .addLayer("fb2_branch_0_0", new BatchNormalization.Builder().build(), "2")
                .addLayer("fb2_branch_0_0_branch_0_0", new BatchNormalization.Builder().build(), "fb2_branch_0_0")
                .addLayer("fb2_branch_0_0_branch_1_0", new BatchNormalization.Builder().build(), "fb2_branch_0_0")
                .addVertex("mergeInner", new ElementWiseVertex(ElementWiseVertex.Op.Add),
                        "fb2_branch_0_0_branch_0_0", "fb2_branch_0_0_branch_1_0")
                .addLayer("fb2_branch_0_1", new BatchNormalization.Builder().build(), "mergeInner")
                .addLayer("fb2_branch_1_0", new BatchNormalization.Builder().build(), "2")
                .addLayer("fb2_branch_2_0", new BatchNormalization.Builder().build(), "2")
                .addVertex("rbMvInput0", new MergeVertex(), "fb2_branch_0_1", "fb2_branch_1_0", "fb2_branch_2_0")
                .addVertex("addMergeAnd1", new ElementWiseVertex(ElementWiseVertex.Op.Add), "rbMvInput0","1")
                .addLayer("3", new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).nOut(7).build(), "addMergeAnd1")
                .addLayer("4", new GlobalPoolingLayer(), "3")
                .addLayer("output", new CenterLossOutputLayer.Builder().nOut(4).build(), "4")
                .build());
        graph.init();
        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 122, 128}));

        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName("1")
                                .mutateNout(nOut -> nOut - 1)
                                .build()))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 122, 128}));
    }

    @Test
    public void tmp() throws IOException {
        final ComputationGraph graph = ModelSerializer.restoreComputationGraph("E:\\Software projects\\java\\leadRythm\\RythmLeadSwitch\\models\\1598149236\\19todebug", true);
        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName("5")
                                .mutateNout(nOut -> nOut - 1)
                                .build()))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 122, 128}));
    }
}