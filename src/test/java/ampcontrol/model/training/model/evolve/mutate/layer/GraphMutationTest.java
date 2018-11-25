package ampcontrol.model.training.model.evolve.mutate.layer;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.Mutation;
import ampcontrol.model.training.model.evolve.mutate.util.CompGraphUtil;
import ampcontrol.model.training.model.evolve.mutate.util.GraphBuilderUtil;
import ampcontrol.model.training.model.layerblocks.*;
import ampcontrol.model.training.model.layerblocks.graph.ForkAgg;
import ampcontrol.model.training.model.layerblocks.graph.ResBlock;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Optional;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;

public class GraphMutationTest {

    /**
     * Test mutation by inserting a layer between two layers
     */
    @Test
    public void mutateInsertSingleLayer() {
        final String mut1 = "mut1";
        final String mut2 = "mut2";
        final String noMut = "noMut";
        final ComputationGraph graph = GraphUtils.getCnnGraph(mut1, mut2, noMut);

        final String toInsert = "between_" + mut1 + "_and_" + mut2;
        final Mutation<ComputationGraphConfiguration.GraphBuilder> mutation = new GraphMutation(() -> Stream.of(
                GraphMutation.GraphMutationDescription.builder()
                        .mutation(graphBuilder -> {
                            graphBuilder.addLayer(toInsert,
                                    new Convolution2D.Builder(5, 5)
                                            .nOut(GraphBuilderUtil.getInputSize(mut2, graphBuilder))
                                            .nIn(GraphBuilderUtil.getOutputSize(mut1, graphBuilder))
                                            .build(), mut1);
                            return GraphMutation.InputsAndOutputNames.builder()
                                    .inputName(mut1)
                                    .keepInputConnection(toInsert::equals)
                                    .outputName(toInsert)
                                    .build();
                        })
                        .build()));
        final ComputationGraph newGraph = new ComputationGraph(mutation.mutate(
                new ComputationGraphConfiguration.GraphBuilder(graph.getConfiguration(), new NeuralNetConfiguration.Builder(graph.conf()))).build());
        newGraph.init();

        assertTrue("Vertex not added!", Optional.ofNullable(newGraph.getVertex(toInsert)).isPresent());
        assertEquals("Incorrect kernel size!", 5,
                newGraph.getLayer(toInsert).getParam(DefaultParamInitializer.WEIGHT_KEY).size(3));

        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }


    /**
     * Test {@link BlockMutationFunction} with a {@link Conv2DBatchNormBefore} followed by a {@link Conv2DBatchNormAfter}.
     */
    @Test
    public void blockMutation() {
        final String mut1 = "mut1";
        final String mut2 = "mut2";
        final String noMut = "noMut";
        final ComputationGraph graph = GraphUtils.getCnnGraph(mut1, mut2, noMut);

        final Function<Long, LayerBlockConfig> toAdd = nOut -> new AggBlock(
                new Conv2DBatchNormBefore()
                        .setConvolutionMode(ConvolutionMode.Same))
                .andThen(new Conv2DBatchNormAfter()
                        .setConvolutionMode(ConvolutionMode.Same)
                        .setNrofKernels(nOut.intValue()));

        final String[] inputNames = new String[]{mut1};
        final Function<String, String> nameMapper = str -> "mutinsert_" + String.join("_", inputNames) + "_" + str;
        final InputType inputType = InputType.convolutional(33, 33, 3);
        final Stream<String> expectedNames = IntStream.range(0, 4)
                .mapToObj(String::valueOf)
                .map(nameMapper);
        final BlockMutationFunction blockMutation = new BlockMutationFunction(
                toAdd,
                inputNames,
                nameMapper
        );
        addBlock(graph, inputType, expectedNames, blockMutation);
    }

    /**
     * Test {@link BlockMutationFunction} with a residual {@link Conv2DBatchNormBefore}
     */
    @Test
    public void resBlockMutation() {
        final String mut1 = "mut1";
        final String mut2 = "mut2";
        final String noMut = "noMut";
        final ComputationGraph graph = GraphUtils.getCnnGraph(mut1, mut2, noMut);

        final Function<Long, LayerBlockConfig> toAdd = nOut -> new ResBlock()
                .setBlockConfig(new Conv2DBatchNormAfter()
                        .setConvolutionMode(ConvolutionMode.Same)
                        .setNrofKernels(nOut.intValue()));

        final Function<String, String> nameMapper = str -> "mutinsert_" + String.join("_", mut1) + "_" + str;
        final Stream<String> expected = Stream.of("0", "1", "rbAdd-1").map(nameMapper);
        addBlock(graph, InputType.convolutional(33, 33, 3), expected,
                new BlockMutationFunction(
                        toAdd,
                        new String[]{mut1},
                        nameMapper));
    }

    /**
     * Test {@link BlockMutationFunction} with a fork of {@link Conv2D} and a {@link Conv2DBatchNormAfter}
     */
    @Test
    public void forkBlockMutation() {
        final String mut1 = "mut1";
        final String mut2 = "mut2";
        final String noMut = "noMut";
        final ComputationGraph graph = GraphUtils.getCnnGraph(mut1, mut2, noMut);

        final Function<Long, LayerBlockConfig> toAdd = nOut -> new ForkAgg()
                .add(new Conv2D()
                        .setConvolutionMode(ConvolutionMode.Same)
                        .setNrofKernels(nOut.intValue() / 2))
                .add(new Conv2DBatchNormAfter()
                        .setConvolutionMode(ConvolutionMode.Same)
                        .setNrofKernels(nOut.intValue() / 2));

        final Function<String, String> nameMapper = str -> "mutinsert_" + String.join("_", mut1) + "_" + str;
        final Stream<String> expected = Stream.of("fb-1_branch_0_0", "fb-1_branch_1_0", "fb-1_branch_1_1").map(nameMapper);
        addBlock(graph, InputType.convolutional(33, 33, 3), expected,
                new BlockMutationFunction(
                        toAdd,
                        new String[]{mut1},
                        nameMapper));
    }

    private void addBlock(ComputationGraph graph, InputType inputType, Stream<String> expectedNames, BlockMutationFunction blockMutation) {
        final Mutation<ComputationGraphConfiguration.GraphBuilder> mutation = new GraphMutation(() -> Stream.of(
                GraphMutation.GraphMutationDescription.builder()
                        .mutation(blockMutation)
                        .build()));
        final ComputationGraph newGraph = new ComputationGraph(mutation.mutate(CompGraphUtil.toBuilder(graph))
                .setInputTypes(inputType)
                .build());
        newGraph.init();

        expectedNames.forEach(expectedName ->
                assertTrue("Vertex " + expectedName + " not added!", Optional.ofNullable(newGraph.getVertex(expectedName)).isPresent()));

        long[] shape = inputType.getShape(true);
        shape[0] = 1;
        graph.outputSingle(Nd4j.randn(shape));
        newGraph.outputSingle(Nd4j.randn(shape));
    }
}