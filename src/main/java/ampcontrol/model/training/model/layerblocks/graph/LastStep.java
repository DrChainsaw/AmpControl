package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adds a {@link LastTimeStepVertex}.
 * 
 * @author Christian Skärby
 */
public class LastStep implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(LastTimeStepVertex.class);

    private String inputName = "input";

    @Override
    public String name() {
        return "ls";
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        throw new IllegalArgumentException("Must work on graphBuilders!");
    }

    @Override
    public BlockInfo addLayers(GraphBuilderAdapter graphBuilder, BlockInfo info) {
        log.info("Last step " + info);
        final int layerInd = info.getPrevLayerInd()+1;
        String thisLayer = info.getName(String.valueOf(layerInd));
        graphBuilder.addVertex(thisLayer, new LastTimeStepVertex(inputName), info.getInputsNames());
        return new SimpleBlockInfo.Builder(info)
                .setPrevLayerInd(layerInd)
                .setInputs(new String[] {thisLayer})
                .build();
    }
}
