package ampcontrol.model.training.model.layerblocks;

import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.Subsampling1DLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.junit.Test;

import java.util.function.Function;

import static org.junit.Assert.assertTrue;

/**
 * Test cases for {@link Pool2D}
 *
 * @author Christian Skärby
 */
public class Pool2DTest {

    /**
     * Test that name reflects set parameters
     */
    @Test
    public void name() {
        final int size = 777;
        final int stride = 888;
        final PoolingType poolingType = PoolingType.MAX;
        String result = new Pool2D()
                .setSize(size)
                .setStride(stride)
                .setType(poolingType)
                .name();
        assertTrue("Parameter not part of name!", result.contains(String.valueOf(size)));
        assertTrue("Parameter not part of name!", result.contains(String.valueOf(stride)));
        assertTrue("Parameter not part of name!", result.contains(poolingType.name().substring(0,1)));
    }

    /**
     * Test addLayers. Tests internals which is bad, but CBA to do anything else for such a simple class.
     */
    @Test
    public void addLayers() {
        final Function<Layer, Boolean> layerChecker = layer -> layer instanceof SubsamplingLayer && !(layer instanceof Subsampling1DLayer);
        final LayerBlockConfig toTest = new Pool2D();
        ProbingBuilderAdapter.testLayerBlock(layerChecker, toTest, 1);
    }
}