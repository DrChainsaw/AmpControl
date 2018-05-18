package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adds a {@link GlobalPoolingLayer}
 *
 * @author Christian Skärby
 */
public class GlobPool implements LayerBlockConfig {
    
    private static final Logger log = LoggerFactory.getLogger(GlobPool.class);

    private PoolingType type = PoolingType.AVG;

    @Override
    public String name() {
        return "gp" + type.name().substring(0,1).toLowerCase();
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        log.info("Global pool " + info);
       return builder.layer(info, new GlobalPoolingLayer.Builder()
               .poolingType(PoolingType.AVG)
               .collapseDimensions(true)
                .build());
    }

    /**
     * Sets the {@link PoolingType}
     * @param type
     * @return the {@link GlobPool}
     */
    public GlobPool setType(PoolingType type) {
        this.type = type;
        return this;
    }
}
