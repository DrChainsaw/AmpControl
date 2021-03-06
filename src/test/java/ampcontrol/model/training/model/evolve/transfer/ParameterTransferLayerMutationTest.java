package ampcontrol.model.training.model.evolve.transfer;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.NoutMutation;
import ampcontrol.model.training.model.evolve.mutate.layer.LayerContainedMutation;
import ampcontrol.model.training.model.evolve.mutate.layer.LayerMutationInfo;
import ampcontrol.model.training.model.evolve.mutate.util.CompGraphUtil;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.distribution.ConstantDistribution;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.graph.ComputationGraph;
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
 * Test cases for {@link ParameterTransfer} with {@link LayerContainedMutation}.
 *
 * @author Christian Skärby
 */
public class ParameterTransferLayerMutationTest {

    /**
     * Test change of kernel size for 2D convolution. One is decreased while the other is increased
     */
    @Test
    public void changeKernelSize() {
        final String mutateName = "toMutate";
        final String mutateTooName = "toMutateToo";
        final ComputationGraph graph = GraphUtils.getCnnGraph(mutateName, mutateTooName, "conv3");

        final int[] orderToKeepFirst = {0, 2, 1};
        final Map<String, Function<Integer, Comparator<Integer>>> comparatorMap = new HashMap<>();
        comparatorMap.put(mutateName, SingleTransferTaskTest.fixedOrderComp(orderToKeepFirst));

        final ParameterTransfer parameterTransfer = new ParameterTransfer(graph,
                name -> Optional.ofNullable(comparatorMap.get(name)));

        final double nextMutationNewVal = 666d;
        final ComputationGraph newGraph = new ComputationGraph(new LayerContainedMutation(() -> Stream.of(
                LayerContainedMutation.LayerMutation.builder()
                        .mutationInfo(
                                LayerMutationInfo.builder()
                                        .layerName(mutateName)
                                        .build())
                        .mutation(layer -> new Convolution2D.Builder(2, 2).build())
                        .build(),
                LayerContainedMutation.LayerMutation.builder()
                        .mutationInfo(
                                LayerMutationInfo.builder()
                                        .layerName(mutateTooName)
                                        .build())
                        .mutation(layer -> new Convolution2D.Builder(4, 4).weightInit(new ConstantDistribution(nextMutationNewVal)).build())
                        .build()))
                .mutate(CompGraphUtil.toBuilder(graph)).build());

        newGraph.init();
        final ComputationGraph mutatedGraph = parameterTransfer.transferWeightsTo(newGraph);

        final INDArray source = graph.getLayer(mutateName).getParam(GraphUtils.W);
        final INDArray target = mutatedGraph.getLayer(mutateName).getParam(GraphUtils.W);
        for (int elemInd = 0; elemInd < target.size(2); elemInd++) {
            assertEquals("Incorrect target for element index " + elemInd + "!",
                    source.tensorAlongDimension(orderToKeepFirst[elemInd], 0, 1, 2).tensorAlongDimension(orderToKeepFirst[elemInd], 0, 1),
                    target.tensorAlongDimension(elemInd, 0, 1, 2).tensorAlongDimension(elemInd, 0, 1));
        }

        final INDArray sourceNext = graph.getLayer(mutateTooName).getParam(GraphUtils.W);
        final INDArray targetNext = mutatedGraph.getLayer(mutateTooName).getParam(GraphUtils.W);
        for (int elemInd = 0; elemInd < sourceNext.size(2); elemInd++) {
            assertEquals("Incorrect target for element index " + elemInd + "!",
                    sourceNext.tensorAlongDimension(elemInd, 0, 1, 2).tensorAlongDimension(elemInd, 0, 1),
                    targetNext.tensorAlongDimension(elemInd, 0, 1, 2).tensorAlongDimension(elemInd, 0, 1));
        }

        final INDArrayIndex[] inds = IntStream.range(0, targetNext.rank()).mapToObj(i -> NDArrayIndex.all()).toArray(INDArrayIndex[]::new);
        inds[2] = NDArrayIndex.interval(sourceNext.size(2), targetNext.size(2));
        inds[3] = NDArrayIndex.interval(sourceNext.size(3), targetNext.size(3));
        assertEquals("Incorrect value!", nextMutationNewVal, targetNext.get(inds).meanNumber().doubleValue(), 1e-10);

        mutatedGraph.output(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    /**
     * Test that a dependent NOut transfer does not block any subsequent kernel size transfers
     */
    @Test
    public void changeKernelSizeAfterNoutChange() {
        final ComputationGraph graph = GraphUtils.getResNet("first", "second", "third");
        ComputationGraphConfiguration.GraphBuilder builder = CompGraphUtil.toBuilder(graph);

        builder = new NoutMutation(() -> Stream.of(
                NoutMutation.NoutMutationDescription.builder()
                        .mutateNout(nOut -> nOut - 1)
                        .layerName("first")
                        .build())).mutate(builder);

        builder = new LayerContainedMutation(() -> Stream.of(
                LayerContainedMutation.LayerMutation.builder()
                        .mutationInfo(
                                LayerMutationInfo.builder()
                                        .layerName("second")
                                        .build())
                        .mutation(layer -> new Convolution2D.Builder(2, 2).convolutionMode(ConvolutionMode.Same).build())
                        .build())).mutate(builder);

        final ComputationGraph newGraph = new ComputationGraph(builder.build());
        newGraph.init();

        new ParameterTransfer(graph).transferWeightsTo(newGraph);

    }
}
