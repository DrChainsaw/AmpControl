package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import org.deeplearning4j.nn.conf.graph.L2NormalizeVertex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adds a {@link L2NormalizeVertex}
 * 
 * @author Christian Skärby
 */
public class Norm implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(Norm.class);
    
    @Override
    public String name() {
        return "N";
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        throw new IllegalArgumentException("Must work with a graphbuilder");
    }

    @Override
    public BlockInfo addLayers(GraphBuilderAdapter graphBuilder, BlockInfo info) {
        log.info("L2 norm: " + info);
        final int layerInd = info.getPrevLayerInd()+1;
        String thisLayer = info.getName(String.valueOf(layerInd));
        graphBuilder.addVertex(thisLayer, new L2NormalizeVertex(), info.getInputsNames());
        return new SimpleBlockInfo.Builder(info)
                .setPrevLayerInd(layerInd)
                .setInputs(new String[] {thisLayer})
                .build();
    }
}
