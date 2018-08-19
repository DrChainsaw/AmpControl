package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adds a {@link SubsamplingLayer}
 *
 * @author Christian Skärby
 */
public class Pool2D implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(Pool2D.class);

    private int size_h = 2;
    private int size_w = 2;
    private int stride_h = 1;
    private int stride_w = 1;
    private PoolingType type = PoolingType.MAX;
    private ConvolutionMode convolutionMode = ConvolutionMode.Truncate;

    @Override
    public String name() {
        String size = size_h == size_w ? "" + size_h : size_h + "_" + size_w;
        String stride = stride_h == stride_w ? "" + stride_h : stride_h + "_" + stride_w;
        String type = this.type.toString().substring(0, 1);
        return type + "p" + size + "_" + stride;
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {

        log.info("P Layer: " + info);
        return builder
                .layer(info, new SubsamplingLayer.Builder(type)
                        .kernelSize(size_h, size_w)
                        .stride(stride_h, stride_w)
                        .convolutionMode(convolutionMode)
                        .build());
    }

    /**
     * Convenience method which sets both pool height and width to the given value
     *
     * @param size
     * @return the {@link Pool2D}
     */
    public Pool2D setSize(int size) {
        setSize_h(size);
        setSize_w(size);
        return this;
    }

    /**
     * Sets pool height to the given value
     *
     * @param size_h
     * @return the {@link Pool2D}
     */
    public Pool2D setSize_h(int size_h) {
        this.size_h = size_h;
        return this;
    }

    /**
     * Sets pool width to the given value
     *
     * @param size_w
     * @return the {@link Pool2D}
     */
    public Pool2D setSize_w(int size_w) {
        this.size_w = size_w;
        return this;
    }

    /**
     * Convenience method which sets both height and width stride to the given value
     *
     * @param stride
     * @return the {@link Pool2D}
     */
    public Pool2D setStride(int stride) {
        setStride_h(stride);
        setStride_w(stride);
        return this;
    }

    /**
     * Sets the height stride
     *
     * @param stride_h
     * @return the {@link Pool2D}
     */
    public Pool2D setStride_h(int stride_h) {
        this.stride_h = stride_h;
        return this;
    }

    /**
     * Sets the width stride
     *
     * @param stride_w
     * @return the {@link Pool2D}
     */
    public Pool2D setStride_w(int stride_w) {
        this.stride_w = stride_w;
        return this;
    }

    /**
     * Sets the {@link PoolingType}
     *
     * @param type
     * @return the {@link Pool2D}
     */
    public Pool2D setType(PoolingType type) {
        this.type = type;
        return this;
    }

    /**
     * Sets the {@link ConvolutionMode} to use
     *
     * @param convolutionMode the mode to use
     * @return the {@link Pool2D}
     */
    public Pool2D setConvolutionMode(ConvolutionMode convolutionMode) {
        this.convolutionMode = convolutionMode;
        return this;
    }
}
