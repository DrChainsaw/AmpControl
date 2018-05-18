package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.Subsampling1DLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adds a {@link Subsampling1DLayer}
 *
 * @author Christian Skärby
 */
public class Pool1D implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(Pool1D.class);

    private int size = 2;
    private int stride = 1;
    private PoolingType type  = PoolingType.MAX;

    @Override
    public String name() {
        String type = this.type.toString().substring(0, 1);
        return  type + "p" + size + "_" + stride;
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {

        log.info("P Layer: " + info);
        return builder
                .layer(info, new Subsampling1DLayer.Builder(type)
                        .kernelSize(size)
                        .stride(stride)
                        .build());
    }

    /**
     * Sets size of the pool
     * @param size
     * @return the {@link Pool1D}
     */
    public Pool1D setSize(int size) {
        this.size = size;
        return this;
    }

    /**
     * Sets stride
     * @param stride
     * @return the {@link Pool1D}
     */
    public Pool1D setStride(int stride) {
        this.stride = stride;
        return this;
    }

    /**
     * Sets {@link PoolingType}
     * @param type
     * @return the {@link Pool1D}
     */
    public Pool1D setType(PoolingType type) {
        this.type = type; return this;
    }
}
