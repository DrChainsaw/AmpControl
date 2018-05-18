package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adds a {@link DropoutLayer}
 * 
 * @author Christian Skärby
 */
public class DropOut implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(DropOut.class);

    private double dropProb = 0.5;

    @Override
    public String name() {
        return "d" + String.valueOf(dropProb).replace('.', 'p');
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        log.info("Dropout " + info);
        return builder.layer(
                info,
                new DropoutLayer.Builder()
                        .dropOut(1-dropProb)
                        .build()
        );
    }

    /**
     * Sets the dropout probability, i.e. the probability that an activation is dropped.
     * @param dropProb
     * @return the {@link DropOut}
     */
    public DropOut setDropProb(double dropProb) {
        this.dropProb = dropProb; return this;
    }

}
