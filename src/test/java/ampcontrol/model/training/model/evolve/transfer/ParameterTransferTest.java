package ampcontrol.model.training.model.evolve.transfer;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.MutateNout;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
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
 * Test cases for {@link ParameterTransfer}
 *
 * @author Christian Skärby
 */
public class ParameterTransferTest {

    /**
     * Test to decrease nOut in a conv layer which is input to another conv layer and see that weights get transferred.
     */
    @Test
    public void decreaseNoutCnnToCnn() {
        final String mutationName = "toMutate";
        final String nextMutationName = "toMutateToo";
        final String afterName = "afterMutate";
        final ComputationGraph graph = GraphUtils.getCnnGraph(mutationName, nextMutationName, afterName);
        final int outputDim = 0;
        final int inputDim = 1;

        final ComputationGraph mutatedGraph = decreaseDecreaseNout(mutationName, nextMutationName, afterName, graph, outputDim, inputDim);

        mutatedGraph.output(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    /**
     * Test one layer increases while the other one increases and see that weights are transferred.
     */
    @Test
    public void IncreaseDecreaseNoutCnnToCnn() {
        final String mutationName = "toMutate";
        final String nextMutationName = "toMutateToo";
        final String afterName = "afterMutate";
        final ComputationGraph graph = GraphUtils.getCnnGraph(mutationName, nextMutationName, afterName);
        final int outputDim = 0;
        final int inputDim = 1;

        final ComputationGraph mutatedGraph = decreaseIncreaseNout(mutationName, nextMutationName, afterName, graph, outputDim, inputDim);

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

        final ComputationGraph mutatedGraph = decreaseDecreaseNout(mutationName, nextMutationName, afterName, graph, outputDim, inputDim);
        mutatedGraph.output(Nd4j.randn(new long[]{1, 33}));
    }

    /**
     * Test one layer increases while the other one increases and see that weights are transferred.
     */
    @Test
    public void IncreaseDecreaseNoutDenseToDense() {
        final String mutationName = "toMutate";
        final String nextMutationName = "toMutateToo";
        final String afterName = "afterMutate";
        final ComputationGraph graph = GraphUtils.getGraph(mutationName, nextMutationName, afterName);
        final int outputDim = 1;
        final int inputDim = 0;

        final ComputationGraph mutatedGraph = decreaseIncreaseNout(mutationName, nextMutationName, afterName, graph, outputDim, inputDim);
        mutatedGraph.output(Nd4j.randn(new long[]{1, 33}));
    }

    @NotNull
    ComputationGraph decreaseDecreaseNout(String mutationName, String nextMutationName, String afterName, ComputationGraph graph, int outputDim, int inputDim) {
        final int[] orderToKeepFirst = {1, 3, 5, 6, 7, 9, 2, 4, 8, 0};
        final int[] orderToKeepSecond = {0, 3, 4, 2, 1};
        final Map<String, Function<int[], Comparator<Integer>>> comparatorMap = new HashMap<>();
        comparatorMap.put(mutationName, SingleTransferTaskTest.fixedOrderComp(orderToKeepFirst));
        comparatorMap.put(nextMutationName, SingleTransferTaskTest.fixedOrderComp(orderToKeepSecond));


        final ParameterTransfer parameterTransfer = new ParameterTransfer(graph,
                name -> Optional.ofNullable(comparatorMap.get(name)));

        final ComputationGraph newGraph = new MutateNout(() -> Stream.of(mutationName, nextMutationName), prevNout -> (int) Math.ceil(prevNout / 2d))
                .mutate(new TransferLearning.GraphBuilder(graph), graph).build();

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
    ComputationGraph decreaseIncreaseNout(String mutationName, String nextMutationName, String afterName, ComputationGraph graph, int outputDim, int inputDim) {
        final int[] orderToKeepFirst = {0, 1, 4, 6, 7, 5, 2, 3, 8, 9};
        final Map<String, Function<int[], Comparator<Integer>>> comparatorMap = new HashMap<>();
        comparatorMap.put(mutationName, SingleTransferTaskTest.fixedOrderComp(orderToKeepFirst));

        final ParameterTransfer parameterTransfer = new ParameterTransfer(graph,
                name -> Optional.ofNullable(comparatorMap.get(name)));

        final int mutationNewNout = 5;
        final int mutationPrevNout = graph.layerSize(mutationName);
        final int nextMutationNewNout = 9;
        final int nextMutationPrevNout = graph.layerSize(nextMutationName);
        final double nextMutationNewVal = 666d; // Is this obtainable somehow?

        final ComputationGraph newGraph = new MutateNout(() -> Stream.of(mutationName, nextMutationName),
                prevNout -> prevNout == mutationPrevNout ? mutationNewNout : prevNout == nextMutationPrevNout ? nextMutationNewNout : -1)
                .mutate(new TransferLearning.GraphBuilder(graph), graph).build();

        final ComputationGraph mutatedGraph = parameterTransfer.transferWeightsTo(newGraph);

        final INDArray source = graph.getLayer(mutationName).getParam(GraphUtils.W);
        final INDArray target = mutatedGraph.getLayer(mutationName).getParam(GraphUtils.W);
        assertDims(outputDim, orderToKeepFirst, source, target);

        final INDArray sourceBias = graph.getLayer(mutationName).getParam(GraphUtils.B);
        final INDArray targetBias = mutatedGraph.getLayer(mutationName).getParam(GraphUtils.B);
        assertDims(1, orderToKeepFirst, sourceBias, targetBias);

        final INDArray sourceNext = graph.getLayer(nextMutationName).getParam(GraphUtils.W);
        final INDArray targetNext = mutatedGraph.getLayer(nextMutationName).getParam(GraphUtils.W);
        assertDoubleDims(IntStream.range(0, nextMutationNewNout).toArray(), orderToKeepFirst, sourceNext, targetNext);
        assertScalar(outputDim, nextMutationPrevNout, nextMutationNewVal, targetNext);

        final INDArray sourceNextBias = graph.getLayer(nextMutationName).getParam(GraphUtils.B);
        final INDArray targetNextBias = mutatedGraph.getLayer(nextMutationName).getParam(GraphUtils.B);
        assertDims(1, IntStream.range(0, nextMutationNewNout).toArray(), sourceNextBias, targetNextBias);
        assertScalar(1, nextMutationPrevNout, 0, targetNextBias);

        final INDArray sourceOutput = graph.getLayer(afterName).getParam(GraphUtils.W);
        final INDArray targetOutput = mutatedGraph.getLayer(afterName).getParam(GraphUtils.W);
        assertDims(inputDim, IntStream.range(0, nextMutationNewNout).toArray(), sourceOutput, targetOutput);
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
            firstTensorDims = new int[] {0};
            secondTensorDims = new int[] {0};
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