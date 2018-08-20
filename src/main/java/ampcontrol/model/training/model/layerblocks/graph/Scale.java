package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import org.deeplearning4j.nn.conf.graph.ScaleVertex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adds a {@link ScaleVertex}
 *
 * @author Christian Skärby
 */
public class Scale implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(Scale.class);

    private final double scaleFactor;

    /** Constructor */
    public Scale(double scaleFactor) {
        this.scaleFactor = scaleFactor;
    }

    @Override
    public String name() {
        return "sc_" + scaleFactor;
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        throw new IllegalArgumentException("Must work with a graphbuilder");
    }

    @Override
    public BlockInfo addLayers(GraphBuilderAdapter graphBuilder, BlockInfo info) {
        log.info("scale: " + info);
        final int layerInd = info.getPrevLayerInd()+1;
        String thisLayer = info.getName(String.valueOf(layerInd));
        graphBuilder.addVertex(thisLayer, new ScaleVertex(scaleFactor), info.getInputsNames());
        return new SimpleBlockInfo.Builder(info)
                .setPrevLayerInd(layerInd)
                .setInputs(new String[] {thisLayer})
                .build();
    }

}
