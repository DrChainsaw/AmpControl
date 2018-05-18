package ampcontrol.model.training.model.layerblocks;

import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.junit.Test;

import java.util.function.Function;

/**
 * Test cases for {@link GlobPool}
 *
 * @author Christian Skärby
 */
public class GlobPoolTest {

    /**
     * Test addLayers. Tests internals which is bad, but CBA to do anything else for such a simple class.
     */
    @Test
    public void addLayers() {
        final Function<Layer, Boolean> layerChecker = layer -> layer instanceof GlobalPoolingLayer;
        final LayerBlockConfig toTest = new GlobPool();
        ProbingBuilderAdapter.testLayerBlock(layerChecker, toTest, 1);
    }

}