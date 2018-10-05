package ampcontrol.model.training.model.evolve.mutate.layer;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.GraphSpyAdapter;
import ampcontrol.model.training.model.layerblocks.graph.SpyBlock;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class BlockMutationFunction implements Function<ComputationGraphConfiguration.GraphBuilder, GraphMutation.InputsAndOutputNames> {

    private final Function<Long, LayerBlockConfig> blockConfigFactory;
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

    BlockMutationFunction(Function<Long, LayerBlockConfig> blockConfigFactory, String[] inputNames, Function<String, String> nameMapping) {
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

        final LayerBlockConfig.BlockInfo outinfo = new SpyBlock(blockConfigFactory.apply(nIn))
                .setFactory(adapter -> new GraphSpyAdapter(spy, adapter))
                .addLayers(graphBuilder, blockInfo);

        return GraphMutation.InputsAndOutputNames.builder()
                .outputName(outinfo.getInputsNames()[0])
                .inputNames(Arrays.asList(inputNames))
                .keepInputConnection(spy.firstLayers::contains)
                .build();
    }
}
