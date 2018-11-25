package ampcontrol.model.training.model.evolve.mutate.layer;

import ampcontrol.model.training.model.evolve.mutate.util.GraphBuilderUtil;
import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.LayerSpyAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.VertexSpyAdapter;
import ampcontrol.model.training.model.layerblocks.graph.SpyBlock;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.HashSet;
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

    private final static class InputLayersSpy implements LayerSpyAdapter.LayerSpy {

        private final Set<String> blockInputs;
        private final Set<String> firstLayers = new HashSet<>();

        private InputLayersSpy(Set<String> blockInputs) {
            this.blockInputs = blockInputs;
        }

        @Override
        public void accept(String layerName, Layer layer, String... layerInputs) {
            if (Stream.of(layerInputs).anyMatch(blockInputs::contains)) {
                firstLayers.add(layerName);
            }
        }
    }

    private final static class InputVerticesSpy implements VertexSpyAdapter.VertexSpy {

        private final Set<String> blockInputs;
        private final Set<String> firstLayers = new HashSet<>();

        private InputVerticesSpy (Set<String> blockInputs) {
            this.blockInputs = blockInputs;
        }

        @Override
        public void accept(String vertexName, GraphVertex vertex, String... vertexInputs) {
            if (Stream.of(vertexInputs).anyMatch(blockInputs::contains)) {
                firstLayers.add(vertexName);
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
                .mapToLong(layerName -> GraphBuilderUtil.getInputSize(layerName, graphBuilder))
                .sum();

        final LayerBlockConfig.BlockInfo blockInfo = new LayerBlockConfig.SimpleBlockInfo.Builder()
                .setInputs(inputNames)
                .setPrevLayerInd(-1)
                .setNameMapper(nameMapping)
                .setPrevNrofOutputs((int) nIn)
                .build();

        final long nOut = Stream.of(inputNames)
                .mapToLong(inputName -> GraphBuilderUtil.getOutputSize(inputName, graphBuilder))
                .sum();

        final InputLayersSpy layersSpy = new InputLayersSpy(Stream.of(inputNames).collect(Collectors.toSet()));
        final InputVerticesSpy verticesSpy = new InputVerticesSpy(Stream.of(inputNames).collect(Collectors.toSet()));


        final LayerBlockConfig conf = new SpyBlock(blockConfigFactory.apply(nOut))
                .setFactory(adapter -> new LayerSpyAdapter(layersSpy, new VertexSpyAdapter(verticesSpy, adapter)));
        log.info("Adding " + conf.name() + " to " + Arrays.toString(inputNames));

        final LayerBlockConfig.BlockInfo outinfo = conf.addLayers(graphBuilder, blockInfo);

        return GraphMutation.InputsAndOutputNames.builder()
                .outputNames(Arrays.asList(outinfo.getInputsNames()))
                .inputNames(Arrays.asList(inputNames))
                .keepInputConnection(name -> layersSpy.firstLayers.contains(name) || verticesSpy.firstLayers.contains(name))
                .build();
    }
}
