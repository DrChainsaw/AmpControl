package ampControl.model.training.model.layerblocks.graph;

import ampControl.model.training.model.layerblocks.LayerBlockConfig;
import ampControl.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampControl.model.training.model.layerblocks.adapters.GraphBuilderAdapter;

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


    private final List<LayerBlockConfig> forkPaths = new ArrayList<>();
    private final String sep;

    public ForkAgg() {
        this( "_b_");
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
                .map(block -> block.name())
                .collect(Collectors.joining(sepToUse)) +"_bf";
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        throw new RuntimeException("Can only do graphs!");
    }

    @Override
    public BlockInfo addLayers(GraphBuilderAdapter graphBuilder, BlockInfo info) {
        final Stream.Builder<String> nextInputNameBuilder = Stream.builder();
        int sumInputSize = 0;
        BlockInfo gpInfo = null;
        System.out.println("Fork block from " + info);
        for(LayerBlockConfig blockConf: forkPaths) {
            blockConf.addLayers(graphBuilder, info);

            Arrays.stream(gpInfo.getInputsNames()).forEach(layerName -> nextInputNameBuilder.accept(layerName));
        }
        // TODO: Mergevertex of StreamBuilder array

        return new SimpleBlockInfo.Builder(gpInfo)
                .setInputs(nextInputNameBuilder.build().collect(Collectors.toList()).toArray(new String[] {}))
                .setPrevNrofOutputs(sumInputSize)
                .build();
    }

    public ForkAgg add(LayerBlockConfig then) {
        forkPaths.add(then);
        return this;
    }
}
