package ampcontrol.model.training.model.layerblocks;

import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.Subsampling1DLayer;
import org.junit.Test;

import java.util.function.Function;

/**
 * Test cases for {@link Pool1D}
 *
 * @author Christian Skärby
 */
public class Pool1DTest {

    /**
     * Test addLayers. Tests internals which is bad, but CBA to do anything else for such a simple class.
     */
    @Test
    public void addLayers() {
        final Function<Layer, Boolean> layerChecker = layer -> layer instanceof Subsampling1DLayer;
        final LayerBlockConfig toTest = new Pool1D();
        ProbingBuilderAdapter.testLayerBlock(layerChecker, toTest, 1);
    }

}