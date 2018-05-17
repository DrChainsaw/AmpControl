package ampcontrol.model.training.model.layerblocks;

import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.junit.Test;

import java.util.function.Function;

/**
 * Test cases for {@link ZeroPad}
 *
 * @author Christian Skärby
 */
public class ZeroPadTest {

    /**
     * Test addLayers. Tests internals which is bad, but CBA to do anything else for such a simple class.
     */
    @Test
    public void addLayers() {
        final Function<Layer, Boolean> layerChecker = layer -> layer instanceof ZeroPaddingLayer;
        final LayerBlockConfig toTest = new ZeroPad();
        ProbingBuilderAdapter.testLayerBlock(layerChecker, toTest, 1);
    }

}