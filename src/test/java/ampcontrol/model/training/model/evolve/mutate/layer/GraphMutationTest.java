package ampcontrol.model.training.model.evolve.mutate.layer;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.Mutation;
import ampcontrol.model.training.model.layerblocks.AggBlock;
import ampcontrol.model.training.model.layerblocks.Conv2DBatchNormAfter;
import ampcontrol.model.training.model.layerblocks.Conv2DBatchNormBefore;
import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.GraphSpyAdapter;
import ampcontrol.model.training.model.layerblocks.graph.SpyBlock;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Optional;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;
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
     * Test {@link BlockMutation} with a {@link Conv2DBatchNormBefore} followed by a {@link Conv2DBatchNormAfter}.
     */
    @Test
    public void blockMutation() {
        final String mut1 = "mut1";
        final String mut2 = "mut2";
        final String noMut = "noMut";
        final ComputationGraph graph = GraphUtils.getCnnGraph(mut1, mut2, noMut);

        final String[] inputNames = new String[]{mut1};
        final Function<String, String> nameMapper = str -> "mutinsert_" + String.join("_", inputNames) + str;
        final BlockMutation blockMutation = new BlockMutation((nIn, nOut) ->
                new AggBlock(
                        new Conv2DBatchNormBefore()
                                .setConvolutionMode(ConvolutionMode.Same))
                        .andThen(new Conv2DBatchNormAfter()
                                .setConvolutionMode(ConvolutionMode.Same)
                                .setNrofKernels(nOut.intValue())),
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

    private static class BlockMutation implements Function<ComputationGraphConfiguration.GraphBuilder, GraphMutation.InputsAndOutputNames> {

        private final BiFunction<Long, Long, LayerBlockConfig> blockConfigFactory;
        private final String[] inputNames;
        private final Function<String, String> nameMapping;

        private final static class FirstLayersSpy implements GraphSpyAdapter.LayerSpy {

            private final Set<String> blockInputs;
            private final Set<String> firstLayers = new HashSet<>();

            private FirstLayersSpy(Set<String> blockInputs) {
                this.blockInputs = blockInputs;
            }

            @Override
            public void accept(String layerName, Layer layer, String... layerInputs) {
                if (Stream.of(layerInputs).anyMatch(blockInputs::contains)) {
                    firstLayers.add(layerName);
                }
            }
        }

        private BlockMutation(BiFunction<Long, Long, LayerBlockConfig> blockConfigFactory, String[] inputNames, Function<String, String> nameMapping) {
            this.blockConfigFactory = blockConfigFactory;
            this.inputNames = inputNames;
            this.nameMapping = nameMapping;
        }

        @Override
        public GraphMutation.InputsAndOutputNames apply(ComputationGraphConfiguration.GraphBuilder graphBuilder) {
            final long nIn = Stream.of(inputNames)
                    .mapToLong(layerName -> LayerMutationInfo.getOutputSize(layerName, graphBuilder))
                    .sum();

            final LayerBlockConfig.BlockInfo blockInfo = new LayerBlockConfig.SimpleBlockInfo.Builder()
                    .setInputs(inputNames)
                    .setPrevLayerInd(-1)
                    .setNameMapper(nameMapping)
                    .setPrevNrofOutputs((int) nIn)
                    .build();

            final FirstLayersSpy spy = new FirstLayersSpy(Stream.of(inputNames).collect(Collectors.toSet()));

            final LayerBlockConfig.BlockInfo outinfo = new SpyBlock(blockConfigFactory.apply(nIn, nIn))
                    .setFactory(adapter -> new GraphSpyAdapter(adapter, spy))
                    .addLayers(graphBuilder, blockInfo);

            return GraphMutation.InputsAndOutputNames.builder()
                    .outputName(outinfo.getInputsNames()[0])
                    .inputNames(Arrays.asList(inputNames))
                    .keepInputConnection(spy.firstLayers::contains)
                    .build();
        }
    }
}