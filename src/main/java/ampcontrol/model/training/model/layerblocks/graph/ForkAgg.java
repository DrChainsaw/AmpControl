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
 * Work in progress {@link LayerBlockConfig} which shall be able to fork and join several
 * {@link LayerBlockConfig LayerBlockConfigs} to create e.g. inception modules.
 *
 * @author Christian Skärby
 */
public class ForkAgg implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(ForkAgg.class);

    private final List<LayerBlockConfig> forkPaths = new ArrayList<>();
    private final String sep;

    public ForkAgg() {
        this( "_f_");
    }

    public ForkAgg(String sep) {
        this.sep = sep;
    }

    @Override
    public String name() {

        String sepToUse = sep;
        if(forkPaths.size() == 1) {
            sepToUse = "";
        }
        return "fb_" + forkPaths.stream()
                .map(LayerBlockConfig::name)
                .collect(Collectors.joining(sepToUse)) +"_bf";
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        throw new UnsupportedOperationException("Can only do graphs!");
    }

    @Override
    public BlockInfo addLayers(GraphBuilderAdapter graphBuilder, BlockInfo info) {
       log.info("ForkAgg block from " + info);

        int cnt = 0;
        int outputSize = 0;
        final Stream.Builder<String> nextInputNameBuilder = Stream.builder();

        for(LayerBlockConfig blockConf: forkPaths) {
            final int branchNr = cnt;
            log.info("ForkAgg path nr " + branchNr + ": " + blockConf.name());
            final BlockInfo branchInInfo = new SimpleBlockInfo.Builder(info)
                    .setNameMapper(name -> info.getName("fb"+info.getPrevLayerInd()+"_branch_" + branchNr + "_" + name))
                    .build();
            final BlockInfo branchOutInfo = blockConf.addLayers(graphBuilder, branchInInfo);
            Arrays.stream(branchOutInfo.getInputsNames()).forEach(nextInputNameBuilder);
            outputSize = branchOutInfo.getPrevNrofOutputs();
            cnt++;
        }

        return new SimpleBlockInfo.Builder(info)
                .setPrevLayerInd(info.getPrevLayerInd()+1)
                .setInputs(nextInputNameBuilder.build().collect(Collectors.toList()).toArray(new String[] {}))
                .setPrevNrofOutputs(outputSize)
                .build();
    }

    public ForkAgg add(LayerBlockConfig then) {
        forkPaths.add(then);
        return this;
    }
}
