package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adds a {@link ZeroPaddingLayer}
 * 
 * @author Christian Skärby
 */
public class ZeroPad implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(ZeroPad.class);

    private int pad_h_top = 3;
    private int pad_w_left = 3;
    private int pad_h_bot = 3;
    private int pad_w_right = 3;

    @Override
    public String name() {
        final String pad_h = pad_h_top == pad_h_bot ? String.valueOf(pad_h_top) : pad_h_top + "_" + pad_h_bot;
        final String pad_w = pad_w_left == pad_w_right ? String.valueOf(pad_w_left) : pad_w_left + "_" + pad_w_right;
        return "zp2_" + pad_h + "_" + pad_w;
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        log.info("Zero pad layer: " + info);
        return builder.layer(
                info,
                new ZeroPaddingLayer.Builder(pad_h_top, pad_h_bot, pad_w_left, pad_w_right)
                        .build());
    }

    /**
     * Convenience method which sets amount of padding in all directions to the same value
     * @param pad
     * @return the {@link ZeroPad}
     */
    public ZeroPad setPad(int pad) {
        setPad_h(pad);
        setPad_w(pad);
        return this;
    }

    /**
     * Convenience method which sets amount of padding in height to the same value
     * @param pad_h
     * @return the {@link ZeroPad}
     */
    public ZeroPad setPad_h(int pad_h) {
        setPad_h_top(pad_h);
        setPad_h_bot(pad_h);
        return this;
    }

    /**
     * Convenience method which sets amount of padding in width to the same value
     * @param pad_w
     * @return the {@link ZeroPad}
     */
    public ZeroPad setPad_w(int pad_w) {
        setPad_w_left(pad_w);
        setPad_w_right(pad_w);
        return this;
    }

    /**
     * Sets amount of padding at the top
     * @param pad_h
     * @return the {@link ZeroPad}
     */
    public ZeroPad setPad_h_top(int pad_h) {
        this.pad_h_top = pad_h;
        return this;
    }

    /**
     * Sets amount of padding at the bottom
     * @param pad_h
     * @return the {@link ZeroPad}
     */
    public ZeroPad setPad_h_bot(int pad_h) {
        this.pad_h_bot = pad_h;
        return this;
    }

    /**
     * Sets amount of padding to the left
     * @param pad_w
     * @return the {@link ZeroPad}
     */
    public ZeroPad setPad_w_left(int pad_w) {
        this.pad_w_left = pad_w;
        return this;
    }

    /**
     * Sets amount of padding to the right
     * @param pad_w
     * @return the {@link ZeroPad}
     */
    public ZeroPad setPad_w_right(int pad_w) {
        this.pad_w_right = pad_w;
        return this;
    }
}
