package ampcontrol.model.training.model.evolve.mutate.layer;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.GraphSpyAdapter;
import ampcontrol.model.training.model.layerblocks.graph.SpyBlock;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Function to use with {@link GraphMutation} which inserts vertices from a supplied {@link LayerBlockConfig} in given
 * {@link ComputationGraphConfiguration.GraphBuilder}s.
 *
 * @author Christian Skärby
 */
public class BlockMutationFunction implements Function<ComputationGraphConfiguration.GraphBuilder, GraphMutation.InputsAndOutputNames> {

    private static final Logger log = LoggerFactory.getLogger(BlockMutationFunction.class);

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

    /**
     * Constructor
     * @param blockConfigFactory Creates {@link LayerBlockConfig}s. Input is size of nOut of the last layer.
     * @param inputNames Names of input vertices
     * @param nameMapping Add stuff to vertex names in order to guarantee uniqueness.
     */
    public BlockMutationFunction(Function<Long, LayerBlockConfig> blockConfigFactory,
                                 String[] inputNames,
                                 Function<String, String> nameMapping) {
        this.blockConfigFactory = blockConfigFactory;
        this.inputNames = inputNames;
        this.nameMapping = nameMapping;
    }

    @Override
    public GraphMutation.InputsAndOutputNames apply(ComputationGraphConfiguration.GraphBuilder graphBuilder) {

        final long nIn = Stream.of(inputNames)
                .mapToLong(layerName -> LayerMutationInfo.getInputSize(layerName, graphBuilder))
                .sum();

        final LayerBlockConfig.BlockInfo blockInfo = new LayerBlockConfig.SimpleBlockInfo.Builder()
                .setInputs(inputNames)
                .setPrevLayerInd(-1)
                .setNameMapper(nameMapping)
                .setPrevNrofOutputs((int) nIn)
                .build();

        final FirstLayersSpy spy = new FirstLayersSpy(Stream.of(inputNames).collect(Collectors.toSet()));

        final long nOut = Stream.of(inputNames)
                .flatMap(inputName -> graphBuilder.getVertexInputs().entrySet()
                        .stream()
                        .filter(entry -> entry.getValue().contains(inputName))
                        .map(Map.Entry::getKey))
                .mapToLong(layerName -> getInputSizeForward(layerName, graphBuilder))
                .sum();

        final LayerBlockConfig conf = new SpyBlock(blockConfigFactory.apply(nOut))
                .setFactory(adapter -> new GraphSpyAdapter(spy, adapter));
        log.info("Adding " + conf.name() + " to " + Arrays.toString(inputNames));

        final LayerBlockConfig.BlockInfo outinfo = conf.addLayers(graphBuilder, blockInfo);

        return GraphMutation.InputsAndOutputNames.builder()
                .outputName(outinfo.getInputsNames()[0])
                .inputNames(Arrays.asList(inputNames))
                .keepInputConnection(spy.firstLayers::contains)
                .build();
    }

    private static long getInputSizeForward(String layerName, ComputationGraphConfiguration.GraphBuilder graphBuilder) {
        return LayerMutationInfo.vertexAsLayerVertex
                .andThen(layerVertex -> LayerMutationInfo.layerVertexAsFeedForward.apply(layerName, layerVertex))
                .apply(layerName, graphBuilder)
                .map(FeedForwardLayer::getNIn)
                .orElseGet(() -> graphBuilder.getVertexInputs().entrySet().stream()
                        .filter(layerToInputsEntry -> layerToInputsEntry.getValue().contains(layerName))
                        .map(Map.Entry::getKey)
                        .mapToLong(inputLayerName -> getInputSizeForward(inputLayerName, graphBuilder))
                        .findAny()
                        .orElseThrow(() -> new IllegalStateException("Could not find any feedforward layers after " + layerName)));
    }
}
