package ampcontrol.model.training.model.evolve.transfer;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.NoutMutation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.ScaleVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.weights.WeightInit;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link ParameterTransfer} with {@link NoutMutation}.
 *
 * @author Christian Skärby
 */
public class ParameterTransferNoutMutationTest {

    /**
     * Test to decrease nOut in a conv layer which is input to another conv layer and see that weights get transferred.
     */
    @Test
    public void decreaseNoutConvToConv() {
        final String mutationName = "toMutate";
        final String nextMutationName = "toMutateToo";
        final String afterName = "afterMutate";
        final ComputationGraph graph = GraphUtils.getCnnGraph(mutationName, nextMutationName, afterName);
        final int outputDim = 0;
        final int inputDim = 1;

        final ComputationGraph mutatedGraph = decreaseDecreaseNout(
                mutationName, nextMutationName, afterName, graph, outputDim, inputDim);

        mutatedGraph.output(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    /**
     * Test one layer increases while the other one increases and see that weights are transferred.
     */
    @Test
    public void increaseDecreaseNoutConvToConv() {
        final String mutationName = "toMutate";
        final String nextMutationName = "toMutateToo";
        final String afterName = "afterMutate";
        final ComputationGraph graph = GraphUtils.getCnnGraph(mutationName, nextMutationName, afterName);
        final int outputDim = 0;
        final int outputDimNext = outputDim;
        final int inputDim = 1;

        final ComputationGraph mutatedGraph = decreaseIncreaseNout(
                mutationName, nextMutationName, afterName, graph, outputDim, outputDimNext, inputDim);

        mutatedGraph.output(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    /**
     * Test to decrease nOut in a dense layer which is input to another dense layer and see that weights get transferred.
     */
    @Test
    public void decreaseNoutDenseToDense() {
        final String mutationName = "toMutate";
        final String nextMutationName = "toMutateToo";
        final String afterName = "afterMutate";
        final ComputationGraph graph = GraphUtils.getGraph(mutationName, nextMutationName, afterName);
        final int outputDim = 1;
        final int inputDim = 0;

        final ComputationGraph mutatedGraph = decreaseDecreaseNout(
                mutationName, nextMutationName, afterName, graph, outputDim, inputDim);
        mutatedGraph.output(Nd4j.randn(new long[]{1, 33}));
    }

    /**
     * Test one layer increases while the other one increases and see that weights are transferred.
     */
    @Test
    public void increaseDecreaseNoutDenseToDense() {
        final String mutationName = "toMutate";
        final String nextMutationName = "toMutateToo";
        final String afterName = "afterMutate";
        final ComputationGraph graph = GraphUtils.getGraph(mutationName, nextMutationName, afterName);
        final int outputDim = 1;
        final int outputDimNext = outputDim;
        final int inputDim = 0;

        final ComputationGraph mutatedGraph = decreaseIncreaseNout(
                mutationName, nextMutationName, afterName, graph, outputDim, outputDimNext, inputDim);
        mutatedGraph.output(Nd4j.randn(new long[]{1, 33}));
    }

    /**
     * Test to decrease nOut in a dense layer which is input to another dense layer and see that weights get transferred.
     */
    @Test
    public void decreaseNoutConvToDense() {
        final String mutationName = "toMutate";
        final String nextMutationName = "toMutateToo";
        final String afterName = "afterMutate";
        final ComputationGraph graph = GraphUtils.getConvToDenseGraph(mutationName, nextMutationName, afterName);
        final int outputDim = 0;
        final int inputDim = 0;

        final ComputationGraph mutatedGraph = decreaseDecreaseNout(
                mutationName, nextMutationName, afterName, graph, outputDim, inputDim);
        mutatedGraph.output(Nd4j.randn(new long[]{1, 3, 9, 9}));
    }

    /**
     * Test one layer increases while the other one increases and see that weights are transferred.
     */
    @Test
    public void increaseDecreaseNoutConvToDense() {
        final String mutationName = "toMutate";
        final String nextMutationName = "toMutateToo";
        final String afterName = "afterMutate";
        final ComputationGraph graph = GraphUtils.getConvToDenseGraph(mutationName, nextMutationName, afterName);
        final int outputDim = 0;
        final int outputDimNext = 1;
        final int inputDim = 0;

        final ComputationGraph mutatedGraph = decreaseIncreaseNout(
                mutationName, nextMutationName, afterName, graph, outputDim, outputDimNext, inputDim);
        mutatedGraph.output(Nd4j.randn(new long[]{1, 3, 9, 9}));
    }

    /**
     * Test to decrease nOut in a graph where a mutated dense layer is input to the output layer
     */
    @Test
    public void decreaseNoutDenseToOut() {
        final String mutationName = "toMutate";
        final String nextMutationName = "toMutateToo";
        final String afterName = "afterMutate";
        final ComputationGraph graph = GraphUtils.getGraphNearOut(mutationName, nextMutationName, afterName);
        final int outputDim = 1;
        final int inputDim = 0;

        final ComputationGraph mutatedGraph = decreaseDecreaseNout(
                mutationName, nextMutationName, afterName, graph, outputDim, inputDim);
        mutatedGraph.output(Nd4j.randn(new long[]{1, 33}));
    }

    /**
     * Test one layer decreases while the other one increases and where the increased one is input to the output layer
     */
    @Test
    public void increaseDecreaseNoutDenseToOut() {
        final String mutationName = "toMutate";
        final String nextMutationName = "toMutateToo";
        final String afterName = "afterMutate";
        final ComputationGraph graph = GraphUtils.getGraphNearOut(mutationName, nextMutationName, afterName);
        final int outputDim = 1;
        final int outputDimNext = outputDim;
        final int inputDim = 0;

        final ComputationGraph mutatedGraph = decreaseIncreaseNout(
                mutationName, nextMutationName, afterName, graph, outputDim, outputDimNext, inputDim);
        mutatedGraph.output(Nd4j.randn(new long[]{1, 33}));
    }

    /**
     * Test to decrease nOut in a residual conv layer followed by batchnorm and another residual conv layer.
     * Smoke test since more or less all layers are touched so checking would be beyond painful.
     */
    @Test
    public void decreaseResNet() {
        final String firstName = "firstResConv";
        final String mutationName = "secondResConvToMutate";
        final String afterName = "afterMutate";
        final ComputationGraph graph = GraphUtils.getResNet(firstName, mutationName, afterName);

        final long newNout = 5;
        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName(mutationName)
                                .mutateNout(nOut -> newNout)
                                .build()))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf()).weightInit(WeightInit.ZERO)))
                .build());
        newGraph.init();
        newGraph.output(Nd4j.randn(new long[]{1, 3, 33, 33}));

        final ComputationGraph mutatedGraph = new ParameterTransfer(graph).transferWeightsTo(newGraph);
        mutatedGraph.output(Nd4j.randn(new long[]{1, 3, 33, 33}));

        Stream.of(mutatedGraph.getVertices())
                .filter(GraphVertex::hasLayer)
                .map(GraphVertex::getLayer)
                .filter(layer -> layer.numParams() > 0)
                .forEach(layer ->
                        assertEquals("Weights not transferred to layer " + layer.conf().getLayer().getLayerName() + "!",
                                graph.getLayer(layer.conf().getLayer().getLayerName()).params().meanNumber(),
                                layer.params().meanNumber())
                );
    }

    /**
     * Test to decrease nOut in conv layer which is merged with another conv layer which is not changed.
     * Expectation is that mapped removed indexes from first conv layer are removed from the layer after
     * the merge vertex.
     */
    @Test
    public void decreaseForkedConv() {
        final String firstName = "firstConv";
        final String afterName = "afterMutate";
        final String fork1NameToMutate = "fork1ToMutate";
        final String fork2Name = "fork2";
        final ComputationGraph graph = GraphUtils.getForkNet(firstName, afterName, fork1NameToMutate, fork2Name);

        final long newNout = 5;
        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName(fork1NameToMutate)
                                .mutateNout(nOut -> newNout)
                                .build()))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();
        newGraph.output(Nd4j.randn(new long[]{1, 3, 33, 33}));

        final int oldNout = graph.layerSize(fork1NameToMutate);

        // Drop the first oldNout - newNout elements from fork1NameToMutate
        final int[] orderToKeep = IntStream.range(0, oldNout)
                .map(i -> (i - (int) newNout))
                .map(i -> i + oldNout)
                .map(i -> i % oldNout).toArray();
        final Map<String, Function<Integer, Comparator<Integer>>> comparatorMap = new HashMap<>();
        comparatorMap.put(fork1NameToMutate, SingleTransferTaskTest.fixedOrderComp(orderToKeep));

        final ComputationGraph mutatedGraph = new ParameterTransfer(graph
                , name -> Optional.ofNullable(comparatorMap.get(name))).transferWeightsTo(newGraph);
        mutatedGraph.output(Nd4j.randn(new long[]{1, 3, 33, 33}));

        final INDArray source = graph.getLayer(fork1NameToMutate).getParam(GraphUtils.W);
        final INDArray target = mutatedGraph.getLayer(fork1NameToMutate).getParam(GraphUtils.W);
        assertDims(0, orderToKeep, source, target);

        final int[] expectedToKeep = IntStream.concat(
                IntStream.of(orderToKeep).limit(newNout),
                IntStream.range(0, graph.layerSize(fork2Name)).map(i -> i + oldNout)).toArray();
        final INDArray sourceAfter = graph.getLayer(afterName).getParam(GraphUtils.W);
        final INDArray targetAfter = mutatedGraph.getLayer(afterName).getParam(GraphUtils.W);
        assertDims(1, expectedToKeep, sourceAfter, targetAfter);
    }


    /**
     * Test to decrease one layer and increase another in a complex graph with merges and element wise operatations
     * Just a smoke test due to pain of checking each individual weight transfer
     */
    @Test
    public void decreaseIncreaseMazeConv() {
        final String firstName = "firstConv";
        final String afterName = "after";
        final String fork1NameToMutate = "fork1ToMutate";
        final String fork2NameToMutate = "fork2ToMutate";
        final String[] fork1Names = {"f1_1", fork1NameToMutate, "f1_3"};
        final String[] fork2Names = {"f2_1", fork2NameToMutate};
        final ComputationGraph graph = GraphUtils.getForkResOuterInnerNet(firstName, afterName, fork1Names, fork2Names);

        graph.output(Nd4j.randn(new long[]{1, 3, 33, 33}));

        final long newNout1 = 5;
        final long newNout2 = 7;
        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName(fork1NameToMutate)
                                .mutateNout(nOut -> newNout1)
                                .build()
//                        ,
//                        NoutMutation.NoutMutationDescription.builder()
//                                .layerName(fork2NameToMutate)
//                                .mutateNout(nOut -> newNout2)
//                                .build()
                ))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())
                        .weightInit(WeightInit.ZERO)))
                .build());
        newGraph.init();
        newGraph.output(Nd4j.randn(new long[]{1, 3, 33, 33}));

        final ComputationGraph mutatedGraph = new ParameterTransfer(graph).transferWeightsTo(newGraph);
        mutatedGraph.output(Nd4j.randn(new long[]{1, 3, 33, 33}));

        Stream.of(mutatedGraph.getVertices())
                .filter(GraphVertex::hasLayer)
                .map(GraphVertex::getLayer)
                .filter(layer -> layer.numParams() > 0)
                .forEach(layer ->
                        assertEquals("Weights not transferred to layer " + layer.conf().getLayer().getLayerName() + "!",
                                graph.getLayer(layer.conf().getLayer().getLayerName()).params().meanNumber(),
                                layer.params().meanNumber())
                );
    }

    /**
     * The trick to not fail this transfer is to not traverse the {@link ElementWiseVertex} twice as this will cause
     * weights for the {@link MergeVertex} to be added twice
     */
    @Test
    public void decreaseInputToElemVertexBeforeFork() {
        final InputType inputType = InputType.convolutional(33, 33, 3);
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(inputType)
                .addInputs("input")
                .setOutputs("output")
                .addLayer("1", new ConvolutionLayer.Builder().nOut(3).convolutionMode(ConvolutionMode.Same).build(),"input" )
                .addVertex("scale1", new ScaleVertex(1), "1")
                .addLayer("2", new BatchNormalization.Builder().nOut(3).build(), "scale1")
                .addLayer("3", new ConvolutionLayer.Builder().nOut(3).convolutionMode(ConvolutionMode.Same).build(), "2")
                .addVertex("add1And3", new ElementWiseVertex(ElementWiseVertex.Op.Add), "1", "3")
                .addLayer("4", new BatchNormalization.Builder().nOut(3).build(), "add1And3")
                .addLayer("5", new BatchNormalization.Builder().nOut(3).build(), "add1And3")
                .addVertex("merge4And5", new MergeVertex(), "4", "5")
                .addLayer("gp", new GlobalPoolingLayer(), "merge4And5")
                .addLayer("output", new CenterLossOutputLayer.Builder().nOut(4).activation(new ActivationSoftmax()).build(), "gp")
                .build());
        graph.init();

        graph.output(Nd4j.randn(new long[]{1, 3, 33, 33}));

        final long newNout = 2;
        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName("1")
                                .mutateNout(nOut -> newNout)
                                .build()
                ))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())
                                        .weightInit(WeightInit.ZERO)))
                .build());
        newGraph.init();
        newGraph.output(Nd4j.randn(new long[]{1, 3, 33, 33}));

        final ComputationGraph mutatedGraph = new ParameterTransfer(graph).transferWeightsTo(newGraph);
        mutatedGraph.output(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }


    @NotNull
    ComputationGraph decreaseDecreaseNout(
            String mutationName, String nextMutationName, String afterName, ComputationGraph graph, int outputDim, int inputDim) {
        final int[] orderToKeepFirst = {1, 3, 5, 6, 7, 9, 2, 4, 8, 0};
        final int[] orderToKeepSecond = {0, 3, 4, 2, 1};
        final Map<String, Function<Integer, Comparator<Integer>>> comparatorMap = new HashMap<>();
        comparatorMap.put(mutationName, SingleTransferTaskTest.fixedOrderComp(orderToKeepFirst));
        comparatorMap.put(nextMutationName, SingleTransferTaskTest.fixedOrderComp(orderToKeepSecond));


        final ParameterTransfer parameterTransfer = new ParameterTransfer(graph,
                name -> Optional.ofNullable(comparatorMap.get(name)));

        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName(mutationName)
                                .mutateNout(nOut -> nOut / 2)
                                .build(),
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName(nextMutationName)
                                .mutateNout(nOut -> nOut / 2)
                                .build()))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        final ComputationGraph mutatedGraph = parameterTransfer.transferWeightsTo(newGraph);
        final INDArray source = graph.getLayer(mutationName).getParam(GraphUtils.W);
        final INDArray target = mutatedGraph.getLayer(mutationName).getParam(GraphUtils.W);
        assertDims(outputDim, orderToKeepFirst, source, target);

        final INDArray sourceBias = graph.getLayer(mutationName).getParam(GraphUtils.B);
        final INDArray targetBias = mutatedGraph.getLayer(mutationName).getParam(GraphUtils.B);
        assertDims(1, orderToKeepFirst, sourceBias, targetBias);

        final INDArray sourceNext = graph.getLayer(nextMutationName).getParam(GraphUtils.W);
        final INDArray targetNext = mutatedGraph.getLayer(nextMutationName).getParam(GraphUtils.W);
        assertDoubleDims(orderToKeepSecond, orderToKeepFirst, sourceNext, targetNext);

        final INDArray sourceNextBias = graph.getLayer(nextMutationName).getParam(GraphUtils.B);
        final INDArray targetNextBias = mutatedGraph.getLayer(nextMutationName).getParam(GraphUtils.B);
        assertDims(1, orderToKeepSecond, sourceNextBias, targetNextBias);

        final INDArray sourceOutput = graph.getLayer(afterName).getParam(GraphUtils.W);
        final INDArray targetOutput = mutatedGraph.getLayer(afterName).getParam(GraphUtils.W);
        assertDims(inputDim, orderToKeepSecond, sourceOutput, targetOutput);
        return mutatedGraph;
    }

    @NotNull
    ComputationGraph decreaseIncreaseNout(
            String mutationName,
            String nextMutationName,
            String afterName,
            ComputationGraph graph,
            int outputDim,
            int outputDimNext,
            int inputDim) {
        final int[] orderToKeepFirst = {0, 1, 4, 6, 7, 5, 2, 3, 8, 9};
        final Map<String, Function<Integer, Comparator<Integer>>> comparatorMap = new HashMap<>();
        comparatorMap.put(mutationName, SingleTransferTaskTest.fixedOrderComp(orderToKeepFirst));

        final ParameterTransfer parameterTransfer = new ParameterTransfer(graph,
                name -> Optional.ofNullable(comparatorMap.get(name)));

        final long mutationNewNout = 5;
        final long nextMutationNewNout = 9;
        final int nextMutationPrevNout = graph.layerSize(nextMutationName);
        final double nextMutationNewVal = 666d; // Is this obtainable somehow?

        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(NoutMutation.NoutMutationDescription.builder()
                                .layerName(mutationName)
                                .mutateNout(nOut -> mutationNewNout)
                                .build(),
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName(nextMutationName)
                                .mutateNout(nOut -> nextMutationNewNout)
                                .build()))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        final ComputationGraph mutatedGraph = parameterTransfer.transferWeightsTo(newGraph);

        final INDArray source = graph.getLayer(mutationName).getParam(GraphUtils.W);
        final INDArray target = mutatedGraph.getLayer(mutationName).getParam(GraphUtils.W);
        assertDims(outputDim, orderToKeepFirst, source, target);

        final INDArray sourceBias = graph.getLayer(mutationName).getParam(GraphUtils.B);
        final INDArray targetBias = mutatedGraph.getLayer(mutationName).getParam(GraphUtils.B);
        assertDims(1, orderToKeepFirst, sourceBias, targetBias);

        final INDArray sourceNext = graph.getLayer(nextMutationName).getParam(GraphUtils.W);
        final INDArray targetNext = mutatedGraph.getLayer(nextMutationName).getParam(GraphUtils.W);
        assertDoubleDims(IntStream.range(0, (int) nextMutationNewNout).toArray(), orderToKeepFirst, sourceNext, targetNext);
        assertScalar(outputDimNext, nextMutationPrevNout, nextMutationNewVal, targetNext);

        final INDArray sourceNextBias = graph.getLayer(nextMutationName).getParam(GraphUtils.B);
        final INDArray targetNextBias = mutatedGraph.getLayer(nextMutationName).getParam(GraphUtils.B);
        assertDims(1, IntStream.range(0, (int) nextMutationNewNout).toArray(), sourceNextBias, targetNextBias);
        assertScalar(1, nextMutationPrevNout, 0, targetNextBias);

        final INDArray sourceOutput = graph.getLayer(afterName).getParam(GraphUtils.W);
        final INDArray targetOutput = mutatedGraph.getLayer(afterName).getParam(GraphUtils.W);
        assertDims(inputDim, IntStream.range(0, (int) nextMutationNewNout).toArray(), sourceOutput, targetOutput);
        return mutatedGraph;
    }

    /**
     * Assert that elements along the given dimension are transferred accordingly
     *
     * @param dim         Dimension which shall be transferred
     * @param orderToKeep Order which elements shall be transferred
     * @param source      {@link INDArray} from which elements shall be transferred
     * @param target      {@link INDArray} to which elements shall be transferred
     */
    private static void assertDims(
            int dim,
            int[] orderToKeep,
            INDArray source,
            INDArray target) {
        final long[] shapeTarget = target.shape();
        final long[] shapeSource = source.shape();
        final int[] dims = IntStream.range(0, shapeTarget.length).filter(i -> i != dim).toArray();
        for (int elemInd = 0; elemInd < Math.min(shapeTarget[dim], shapeSource[dim]); elemInd++) {
            assertEquals("Incorrect target for element index " + elemInd + "!",
                    source.tensorAlongDimension(orderToKeep[elemInd], dims),
                    target.tensorAlongDimension(elemInd, dims));
        }
    }

    /**
     * Assert that mean value of all elements after a certain element for a dimension is equal to an expected value.
     *
     * @param dim      Dimension for mean value
     * @param start    Mean value of all elements after this one along dim shall be computed
     * @param expected Expected mean value
     * @param actual   {@link INDArray} which mean value shall be computed for
     */
    private static void assertScalar(int dim, int start, double expected, INDArray actual) {
        final INDArrayIndex[] inds = IntStream.range(0, actual.rank()).mapToObj(i -> NDArrayIndex.all()).toArray(INDArrayIndex[]::new);
        inds[dim] = NDArrayIndex.interval(start, actual.size(dim));
        assertEquals("Incorrect value!", expected, actual.get(inds).meanNumber().doubleValue(), 1e-10);
    }

    /**
     * Assert that elements of dimension 0 and dimension 1 are transferred according to a certain order. Typical use
     * case is when targets nOut has changed as the result of a mutation while its nIn has changed as a result of an
     * input layers mutation.
     *
     * @param expectedElementOrderDim0 Expected order for dim 0
     * @param expectedElementOrderDim1 Expected order for dim 1
     * @param source                   {@link INDArray} from which elements shall have been transferred to target
     * @param target                   {@link INDArray} to which elements shall have been transferred from source
     */
    private static void assertDoubleDims(
            int[] expectedElementOrderDim0,
            int[] expectedElementOrderDim1,
            INDArray source,
            INDArray target) {
        final long[] shapeTarget = target.shape();
        final long[] shapeSource = source.shape();
        final int[] firstTensorDims;
        final int[] secondTensorDims;
        final int outputDim;
        final int inputDim;
        if (shapeSource.length > 2) {
            firstTensorDims = IntStream.range(0, shapeTarget.length).filter(i -> i != 0).toArray();
            secondTensorDims = IntStream.range(0, Math.max(1, shapeTarget.length - 2)).map(i -> i + 1).toArray();
            outputDim = 0;
            inputDim = 1;
        } else {
            firstTensorDims = new int[]{0};
            secondTensorDims = new int[]{0};
            outputDim = 1;
            inputDim = 0;
        }

        for (int elemInd0 = 0; elemInd0 < Math.min(shapeTarget[outputDim], shapeSource[outputDim]); elemInd0++) {
            for (int elemInd1 = 0; elemInd1 < Math.min(shapeTarget[inputDim], shapeSource[inputDim]); elemInd1++) {
                assertEquals("Incorrect target output for element index " + elemInd0 + ", " + elemInd1 + "!",
                        source.tensorAlongDimension(expectedElementOrderDim0[elemInd0], firstTensorDims).tensorAlongDimension(expectedElementOrderDim1[elemInd1], secondTensorDims),
                        target.tensorAlongDimension(elemInd0, firstTensorDims).tensorAlongDimension(elemInd1, secondTensorDims));
            }
        }
    }

}