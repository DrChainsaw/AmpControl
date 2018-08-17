package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampcontrol.model.training.model.layerblocks.graph.SeBlock;
import org.deeplearning4j.nn.conf.layers.ZeroPadding1DLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Adds a {@link ZeroPadding1DLayer}.
 * 
 * @author Christian Skärby
 */
public class ZeroPad1D implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(SeBlock.class);

    private int paddingLeft = 0;
    private int paddingRight = 0;
    
    @Override
    public String name() {
        String padStr = paddingLeft == paddingRight ? ""+paddingLeft : paddingLeft + "_" + paddingRight;
        return "zp1_" + padStr;
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        log.info("Zero pad 1D " + info);
        return builder.layer(
                info,
                new ZeroPadding1DLayer.Builder(paddingLeft, paddingRight)
                        .build());
    }

    /**
     * Sets the amount of padding in both left and right direction
     * @param padding
     * @return the {@link ZeroPad1D}
     */
    public ZeroPad1D setPadding(int padding) {
        this.paddingLeft = padding;
        this.paddingRight = padding; return this;
    }

    /**
     * Sets the amount of padding in left direction
     * @param paddingLeft
     * @return the {@link ZeroPad1D}
     */
    public ZeroPad1D setPaddingLeft(int paddingLeft) {
        this.paddingLeft = paddingLeft; return this;
    }

    /**
     * Sets the amount of padding in right direction
     * @param paddingRight
     * @return the {@link ZeroPad1D}
     */
    public ZeroPad1D setPaddingRight(int paddingRight) {
        this.paddingRight = paddingRight; return this;
    }
}
