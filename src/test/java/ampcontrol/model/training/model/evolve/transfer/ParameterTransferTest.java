package ampcontrol.model.training.model.evolve.transfer;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.MutateNout;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.junit.Test;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
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

    final MemoryWorkspace workspace = Nd4j.getWorkspaceManager().createNewWorkspace(WorkspaceConfiguration.builder()
                    .policyAllocation(AllocationPolicy.STRICT)
                    .policyLearning(LearningPolicy.FIRST_LOOP)
                    .policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                    .policySpill(SpillPolicy.REALLOCATE)
                    .initialSize(0)
                    .build(),
            this.getClass().getSimpleName() + "Workspace" + this.toString().split("@")[1]);

    /**
     * Test to decrease nOut in a CNN layer which is input to another CNN layer and see that weights get transferred.
     */
    @Test
    public void decreaseNoutCnnToCnn() {
        final String mutationName = "toMutate";
        final String nextMutationName = "toMutateToo";
        final String afterName = "afterMutate";
        final ComputationGraph graph = GraphUtils.getCnnGraph(mutationName, nextMutationName, afterName);

        final int[] orderToKeepFirst = {1, 3, 5, 6, 7, 9, 2, 4, 8, 0};
        final int[] orderToKeepSecond = {0, 3, 4, 2, 1};
        final Map<String, Function<int[], Comparator<Integer>>> comparatorMap = new HashMap<>();
        comparatorMap.put(mutationName, SingleTransferTaskTest.fixedOrderComp(orderToKeepFirst));
        comparatorMap.put(nextMutationName, SingleTransferTaskTest.fixedOrderComp(orderToKeepSecond));

        ComputationGraph mutatedGraph;
        try(MemoryWorkspace ws = workspace.notifyScopeEntered()) {
            final ParameterTransfer parameterTransfer = new ParameterTransfer(graph,
                    name -> Optional.ofNullable(comparatorMap.get(name)));

            final ComputationGraph newGraph = new MutateNout(() -> Stream.of(mutationName, nextMutationName), prevNout -> (int) Math.ceil(prevNout / 2d))
                    .mutate(new TransferLearning.GraphBuilder(graph), graph).build();

            mutatedGraph = parameterTransfer.transferWeightsTo(newGraph);
        }
        final INDArray source = graph.getLayer(mutationName).getParam(GraphUtils.W);
        final INDArray target = mutatedGraph.getLayer(mutationName).getParam(GraphUtils.W);
        assertDims(0, orderToKeepFirst, source, target);

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
        assertDims(1, orderToKeepSecond, sourceOutput, targetOutput);

        mutatedGraph.output(Nd4j.randn(new long[] {1 ,3, 33, 33}));
    }

    /**
     * Test one layer increases while the other one increases and see that weights are transferred.
     */
    @Test
    public void decreaseIncreaseCnnToCnn() {
        final String mutationName = "toMutate";
        final String nextMutationName = "toMutateToo";
        final String afterName = "afterMutate";
        final ComputationGraph graph = GraphUtils.getCnnGraph(mutationName, nextMutationName, afterName);

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

        ComputationGraph mutatedGraph;
        try(MemoryWorkspace ws = workspace.notifyScopeEntered()) {
            final ComputationGraph newGraph = new MutateNout(() -> Stream.of(mutationName, nextMutationName),
                    prevNout -> prevNout == mutationPrevNout ? mutationNewNout : prevNout == nextMutationPrevNout ? nextMutationNewNout : -1)
                    .mutate(new TransferLearning.GraphBuilder(graph), graph).build();

            mutatedGraph = parameterTransfer.transferWeightsTo(newGraph);
        }

        final INDArray source = graph.getLayer(mutationName).getParam(GraphUtils.W);
        final INDArray target = mutatedGraph.getLayer(mutationName).getParam(GraphUtils.W);
        assertDims(0, orderToKeepFirst, source, target);

        final INDArray sourceBias = graph.getLayer(mutationName).getParam(GraphUtils.B);
        final INDArray targetBias = mutatedGraph.getLayer(mutationName).getParam(GraphUtils.B);
        assertDims(1, orderToKeepFirst, sourceBias, targetBias);

        final INDArray sourceNext = graph.getLayer(nextMutationName).getParam(GraphUtils.W);
        final INDArray targetNext = mutatedGraph.getLayer(nextMutationName).getParam(GraphUtils.W);
        assertDoubleDims(IntStream.range(0, nextMutationNewNout).toArray(), orderToKeepFirst, sourceNext, targetNext);
        assertScalar(0, nextMutationPrevNout, nextMutationNewVal, targetNext);

        final INDArray sourceNextBias = graph.getLayer(nextMutationName).getParam(GraphUtils.B);
        final INDArray targetNextBias = mutatedGraph.getLayer(nextMutationName).getParam(GraphUtils.B);
        assertDims(1, IntStream.range(0, nextMutationNewNout).toArray(), sourceNextBias, targetNextBias);
        assertScalar(1, nextMutationPrevNout, 0, targetNextBias);

        final INDArray sourceOutput = graph.getLayer(afterName).getParam(GraphUtils.W);
        final INDArray targetOutput = mutatedGraph.getLayer(afterName).getParam(GraphUtils.W);
        assertDims(1, IntStream.range(0, nextMutationNewNout).toArray(), sourceOutput, targetOutput);

        mutatedGraph.output(Nd4j.randn(new long[] {1 ,3, 33, 33}));

    }

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

    private static void assertScalar(int dim, int start, double expected, INDArray actual) {
        final INDArrayIndex[] inds = IntStream.range(0, actual.rank()).mapToObj(i -> NDArrayIndex.all()).toArray(INDArrayIndex[]::new);
        inds[dim] = NDArrayIndex.interval(start, actual.size(dim));
        assertEquals("Incorrect value!", expected , actual.get(inds).meanNumber().doubleValue(), 1e-10);
    }

    private static void assertDoubleDims(
            int[] expectedElementOrderDim0,
            int[] expectedElementOrderDim1,
            INDArray source,
            INDArray target) {
        final long[] shapeTarget = target.shape();
        final long[] shapeSource = source.shape();
        final int[] firstTensorDims = IntStream.range(0, shapeTarget.length).filter(i -> i != 0).toArray();
        final int[] secondTensorDims = IntStream.range(0, shapeTarget.length-2).map(i -> i+1).toArray();
        for (int elemInd0 = 0; elemInd0 < Math.min(shapeTarget[0], shapeSource[0]); elemInd0++) {
            for (int elemInd1 = 0; elemInd1 < Math.min(shapeTarget[1], shapeSource[1]); elemInd1++) {
                assertEquals("Incorrect target output for element index " + elemInd0 + ", " + elemInd1 + "!",
                        source.tensorAlongDimension(expectedElementOrderDim0[elemInd0], firstTensorDims).tensorAlongDimension(expectedElementOrderDim1[elemInd1],secondTensorDims),
                        target.tensorAlongDimension(elemInd0, firstTensorDims).tensorAlongDimension(elemInd1, secondTensorDims));
            }
        }
    }

}