package ampcontrol.model.training.model.evolve.mutate.layer;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.Mutation;
import ampcontrol.model.training.model.layerblocks.AggBlock;
import ampcontrol.model.training.model.layerblocks.Conv2DBatchNormAfter;
import ampcontrol.model.training.model.layerblocks.Conv2DBatchNormBefore;
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
import static org.junit.Assert.assertFalse;

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
        final Mutation<ComputationGraphConfiguration.GraphBuilder> mutatation = new GraphMutation(() -> Stream.of(
                GraphMutation.GraphMutationDescription.builder()
                        .mutation(graphBuilder -> {
                            graphBuilder.addLayer(toInsert,
                                    new Convolution2D.Builder(5, 5)
                                            .nOut(LayerMutationInfo.getInputSize(mut2, graphBuilder))
                                            .nIn(LayerMutationInfo.getOutputSize(mut1, graphBuilder))
                                            .build(), mut1);
                            return GraphMutation.InputsAndOutputNames.builder()
                                    .inputName(mut1)
                                    .keepInputConnection(toInsert::equals)
                                    .outputName(toInsert)
                                    .build();
                        })
                        .build()));
        final ComputationGraph newGraph = new ComputationGraph(mutatation.mutate(
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

        final String[] inputNames = new String[]{mut1};
        final Function<String, String> nameMapper = str -> "mutinsert_" + String.join("_", inputNames) + str;
        final BlockMutationFunction blockMutation = new BlockMutationFunction((nIn) ->
                new AggBlock(
                        new Conv2DBatchNormBefore()
                                .setConvolutionMode(ConvolutionMode.Same))
                        .andThen(new Conv2DBatchNormAfter()
                                .setConvolutionMode(ConvolutionMode.Same)
                                .setNrofKernels(nIn.intValue())),
                inputNames,
                nameMapper
        );
        final Mutation<ComputationGraphConfiguration.GraphBuilder> mutatation = new GraphMutation(() -> Stream.of(
                GraphMutation.GraphMutationDescription.builder()
                        .mutation(blockMutation)
                        .build()));
        final ComputationGraph newGraph = new ComputationGraph(mutatation.mutate(
                new ComputationGraphConfiguration.GraphBuilder(graph.getConfiguration(), new NeuralNetConfiguration.Builder(graph.conf())))
                .setInputTypes(InputType.convolutional(33, 33, 3))
                .build());
        newGraph.init();

        IntStream.range(0, 4)
                .mapToObj(String::valueOf)
                .map(nameMapper)
                .forEach(expectedName ->
                        assertTrue("Vertex " + expectedName + "not added!", Optional.ofNullable(newGraph.getVertex(expectedName)).isPresent()));

        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    /**
     * Test removal of a convolution layer.
     */
    @Test
    public void removeConvVertex() {
        final String conv1 = "conv1";
        final String conv2 = "conv2";
        final String conv3 = "conv3";
        final ComputationGraph graph = GraphUtils.getCnnGraph(conv1, conv2, conv3);

        final Mutation<ComputationGraphConfiguration.GraphBuilder> mutatation = new GraphMutation(() -> Stream.of(
                GraphMutation.GraphMutationDescription.builder()
                        .mutation(new RemoveLayerFunction(conv2))
                        .build()));
        final ComputationGraph newGraph = new ComputationGraph(mutatation.mutate(
                new ComputationGraphConfiguration.GraphBuilder(graph.getConfiguration(), new NeuralNetConfiguration.Builder(graph.conf())))
                .setInputTypes(InputType.convolutional(33, 33, 3))
                .build());
        newGraph.init();

        assertFalse("Expected vertex to be removed!", newGraph.getConfiguration().getVertices().containsKey(conv2));

        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
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

        final Mutation<ComputationGraphConfiguration.GraphBuilder> mutatation = new GraphMutation(() -> Stream.of(
                GraphMutation.GraphMutationDescription.builder()
                        .mutation(new RemoveLayerFunction(dense1))
                        .build()));
        final ComputationGraph newGraph = new ComputationGraph(mutatation.mutate(
                new ComputationGraphConfiguration.GraphBuilder(graph.getConfiguration(), new NeuralNetConfiguration.Builder(graph.conf())))
                .setInputTypes(InputType.feedForward(33))
                .build());
        newGraph.init();

        assertFalse("Expected vertex to be removed!", newGraph.getConfiguration().getVertices().containsKey(dense1));

        graph.outputSingle(Nd4j.randn(new long[]{1, 33}));
        newGraph.outputSingle(Nd4j.randn(new long[]{1, 33}));
    }

}