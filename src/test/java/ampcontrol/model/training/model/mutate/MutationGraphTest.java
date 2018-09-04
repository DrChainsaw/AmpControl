package ampcontrol.model.training.model.mutate;

import ampcontrol.model.training.model.mutate.reshape.SingleTransferTaskTest;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.ConstantDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;
import java.util.function.Function;
import java.util.stream.IntStream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link MutationGraph}
 *
 * @author Christian Skärby
 */
public class MutationGraphTest {

    private final static String inputName = "input";
    private final static String outputName = "output";

    private final static String W = DefaultParamInitializer.WEIGHT_KEY;
    private final static String B = DefaultParamInitializer.BIAS_KEY;


    /**
     * Test to decrease nOut in a CNN layer which is input to another CNN layer and see that weights get transferred
     */
    @Test
    public void decreaseNoutCnnToCnn() {
        final String mutationName = "toMutate";
        final String nextMutationName = "toMutateToo";
        final String afterName = "afterMutate";
        final ComputationGraph graph = getCnnGraph(mutationName, nextMutationName, afterName);

        final int[] orderToKeepFirst = {1, 3, 5, 6, 7, 9, 2, 4, 8, 0};
        final int[] orderToKeepSecond = {0, 3, 4, 2, 1};
        final Map<String, Function<int[], Comparator<Integer>>> comparatorMap = new HashMap<>();
        comparatorMap.put(mutationName, SingleTransferTaskTest.fixedOrderComp(orderToKeepFirst));
        comparatorMap.put(nextMutationName, SingleTransferTaskTest.fixedOrderComp(orderToKeepSecond));

        final MutationGraph mutationGraph = new MutationGraph(graph,
                name -> Optional.ofNullable(comparatorMap.get(name)));

        // Workaround for dl4j issue #6343. Create the graphs step by step to avoid overwriting nIn to nextMutationName
         final ComputationGraph newGraph = new TransferLearning.GraphBuilder(new TransferLearning
                .GraphBuilder(graph)
                .nOutReplace(mutationName, 5, new ConstantDistribution(333))
                .build())
          .nOutReplace(nextMutationName, 3, new ConstantDistribution(111))
         .build();

        assertEquals("newGraph not initialized as expected!",
                333d,
                newGraph.getLayer(mutationName).paramTable().get(W).meanNumber().doubleValue(), 1e-10);

        assertEquals("newGraph not initialized as expected!",
                111d,
                newGraph.getLayer(nextMutationName).paramTable().get(W).meanNumber().doubleValue(), 1e-10);

        assertEquals("newGraph not initialized as expected!",
                111d,
                newGraph.getLayer(afterName).paramTable().get(W).meanNumber().doubleValue(), 1e-10);


        final ComputationGraph mutatedGraph = mutationGraph.mutateTo(newGraph.getConfiguration());

        final INDArray source = graph.getLayer(mutationName).getParam(W);
        final INDArray target = mutatedGraph.getLayer(mutationName).getParam(W);
        assertDims(0, orderToKeepFirst, source, target);

        final INDArray sourceBias = graph.getLayer(mutationName).getParam(B);
        final INDArray targetBias = mutatedGraph.getLayer(mutationName).getParam(B);
        assertDims(1, orderToKeepFirst, sourceBias, targetBias);

        final INDArray sourceNext = graph.getLayer(nextMutationName).getParam(W);
        final INDArray targetNext = mutatedGraph.getLayer(nextMutationName).getParam(W);
        assertDoubleDims(orderToKeepSecond, orderToKeepFirst, sourceNext, targetNext);

        final INDArray sourceNextBias = graph.getLayer(nextMutationName).getParam(B);
        final INDArray targetNextBias = mutatedGraph.getLayer(nextMutationName).getParam(B);
        assertDims(1, orderToKeepSecond, sourceNextBias, targetNextBias);

        final INDArray sourceOutput = graph.getLayer(afterName).getParam(W);
        final INDArray targetOutput = mutatedGraph.getLayer(afterName).getParam(W);
        assertDims(1, orderToKeepSecond, sourceOutput, targetOutput);
    }

    @Test
    public void decreaseIncreaseCnnToCnn() {
        final String mutationName = "toMutate";
        final String nextMutationName = "toMutateToo";
        final String afterName = "afterMutate";
        final ComputationGraph graph = getCnnGraph(mutationName, nextMutationName, afterName);

        final int[] orderToKeepFirst = {0, 1, 4, 6, 7, 5, 2, 3, 8, 9};
        final Map<String, Function<int[], Comparator<Integer>>> comparatorMap = new HashMap<>();
        comparatorMap.put(mutationName, SingleTransferTaskTest.fixedOrderComp(orderToKeepFirst));

        final MutationGraph mutationGraph = new MutationGraph(graph,
                name -> Optional.ofNullable(comparatorMap.get(name)));

        final int nextMutationNewNout = 9;
        final int nextMutationPrevNout = 5;
        final double nextMutationNewVal = 111d;
        // Workaround for dl4j issue #6343. Create the graphs step by step to avoid overwriting nIn to nextMutationName
        final ComputationGraph newGraph = new TransferLearning.GraphBuilder(new TransferLearning
                .GraphBuilder(graph)
                .nOutReplace(mutationName, 5, new ConstantDistribution(333))
                .build())
                .nOutReplace(nextMutationName, nextMutationNewNout, new ConstantDistribution(nextMutationNewVal))
                .build();


        final ComputationGraph mutatedGraph = mutationGraph.mutateTo(newGraph.getConfiguration());

        final INDArray source = graph.getLayer(mutationName).getParam(W);
        final INDArray target = mutatedGraph.getLayer(mutationName).getParam(W);
        assertDims(0, orderToKeepFirst, source, target);

        final INDArray sourceBias = graph.getLayer(mutationName).getParam(B);
        final INDArray targetBias = mutatedGraph.getLayer(mutationName).getParam(B);
        assertDims(1, orderToKeepFirst, sourceBias, targetBias);

        final INDArray sourceNext = graph.getLayer(nextMutationName).getParam(W);
        final INDArray targetNext = mutatedGraph.getLayer(nextMutationName).getParam(W);
        assertDoubleDims(IntStream.range(0, nextMutationNewNout).toArray(), orderToKeepFirst, sourceNext, targetNext);
        assertScalar(0, nextMutationPrevNout, nextMutationNewVal, targetNext);

        final INDArray sourceNextBias = graph.getLayer(nextMutationName).getParam(B);
        final INDArray targetNextBias = mutatedGraph.getLayer(nextMutationName).getParam(B);
        assertDims(1, IntStream.range(0, nextMutationNewNout).toArray(), sourceNextBias, targetNextBias);
        assertScalar(1, nextMutationPrevNout, 0, targetNextBias);

        final INDArray sourceOutput = graph.getLayer(afterName).getParam(W);
        final INDArray targetOutput = mutatedGraph.getLayer(afterName).getParam(W);
        assertDims(1, IntStream.range(0, nextMutationNewNout).toArray(), sourceOutput, targetOutput);

    }

    @NotNull
    static ComputationGraph getCnnGraph(String mutationName, String nextMutationName, String afterName) {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .weightInit(new ConstantDistribution(666))
                .graphBuilder()
                .addInputs(inputName)
                .setOutputs(outputName)
                .setInputTypes(InputType.convolutional(33, 33, 3))
                .addLayer(mutationName, new Convolution2D.Builder(3, 3)
                        .nOut(10)
                        .build(), inputName)
                .addLayer(nextMutationName, new Convolution2D.Builder(1, 1)
                        .nOut(5)
                        .build(), mutationName)
                .addLayer(afterName, new Convolution2D.Builder(2, 2)
                        .nOut(7)
                        .build(), nextMutationName)
                .addLayer(outputName, new OutputLayer.Builder()
                        .nOut(2)
                        .build(), afterName)
                .build());
        graph.init();

        setToLinspace(graph.getLayer(mutationName).getParam(W), true);
        setToLinspace(graph.getLayer(nextMutationName).getParam(W), false);
        setToLinspace(graph.getLayer(afterName).getParam(W), true);
        setToLinspace(graph.getLayer(mutationName).getParam(B), false);
        setToLinspace(graph.getLayer(nextMutationName).getParam(B), true);
        return graph;
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

    private static void setToLinspace(INDArray array, boolean reverse) {
        final long nrofElemsSource = Arrays.stream(array.shape()).reduce((i1, i2) -> i1 * i2).orElseThrow(() -> new IllegalArgumentException("No elements!"));
        array.assign(Nd4j.linspace(0, nrofElemsSource - 1, nrofElemsSource).reshape(array.shape()));
        if (reverse) {
            Nd4j.reverse(array);
        }
    }
}