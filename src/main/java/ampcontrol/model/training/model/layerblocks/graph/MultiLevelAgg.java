package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Multi level feature aggregation for use in training. Is configured with a sequence of {@link LayerBlockConfig 
 * LayerBlockConfigs} where output from each element in the sequence will be used as input to the next element, just as
 * in a normal multilayer network. The difference is that the {@link GlobMeanMax} of the individual outputs from all
 * elements in the sequence will be concatenated and used as input to the next {@link LayerBlockConfig}.
 * 
 * Original idea: https://arxiv.org/abs/1703.01793
 * This incarnation which can also be trained: https://arxiv.org/abs/1710.10451
 * 
 * @author Christian Skärby
 */
public class MultiLevelAgg implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(MultiLevelAgg.class);

    private final List<LayerBlockConfig> sequence = new ArrayList<>();
    private final String sep;

    public MultiLevelAgg() {
        this( "_t_");
    }

    public MultiLevelAgg(String sep) {
        this.sep = sep;
    }

    @Override
    public String name() {

        String sepToUse = sep;
        if(sequence.size() == 1) {
            sepToUse = "";
        }
        return "mla_" + sequence.stream()
                .map(block -> block.name())
                .collect(Collectors.joining(sepToUse)) +"_alm";
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        final Stream.Builder<String> nextInputNameBuilder = Stream.builder();
        int sumInputSize = 0;
        BlockInfo nextInfo = info;
        for(LayerBlockConfig blockConf: sequence) {
            nextInfo = blockConf.addLayers(builder, nextInfo);
            Arrays.stream(nextInfo.getInputsNames()).forEach(layerName -> nextInputNameBuilder.accept(layerName));
            sumInputSize += nextInfo.getPrevNrofOutputs();
        }
        return new SimpleBlockInfo.Builder(nextInfo)
                .setInputs(nextInputNameBuilder.build().collect(Collectors.toList()).toArray(new String[] {}))
                .setPrevNrofOutputs(sumInputSize)
                .build();
    }

    @Override
    public BlockInfo addLayers(GraphBuilderAdapter graphBuilder, BlockInfo info) {
        final Stream.Builder<String> nextInputNameBuilder = Stream.builder();
        int sumInputSize = 0;
        BlockInfo nextInfo = info;
        BlockInfo gpInfo = null;
        log.info("MultiLevel block from " + info);
        for(LayerBlockConfig blockConf: sequence) {
            nextInfo = blockConf.addLayers(graphBuilder, nextInfo);
            log.info("Multilevel nextInfo: " + nextInfo);

            gpInfo = new GlobMeanMax().addLayers(graphBuilder, nextInfo);

            Arrays.stream(gpInfo.getInputsNames()).forEach(layerName -> nextInputNameBuilder.accept(layerName));
            sumInputSize += nextInfo.getPrevNrofOutputs();
        }
        return new SimpleBlockInfo.Builder(gpInfo)
                .setInputs(nextInputNameBuilder.build().collect(Collectors.toList()).toArray(new String[] {}))
                .setPrevNrofOutputs(sumInputSize)
                .build();
    }

    /**
     * Adds a {@link LayerBlockConfig} to the sequence which shall be multi level aggregated.
     * @param then
     * @return the {@link MultiLevelAgg}
     */
    public MultiLevelAgg andThen(LayerBlockConfig then) {
        sequence.add(then);
        return this;
    }
}
