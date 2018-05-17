package ampcontrol.model.training.model.layerblocks;

import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.junit.Test;

import java.util.function.Function;

/**
 * Test case for {@link Act}
 *
 * @author Christian Skärby
 */
public class ActTest {

    /**
     * Test addLayers. Tests internals which is bad, but CBA to do anything else for such a simple class.
     */
    @Test
    public void addLayers() {
        final Function<Layer, Boolean> layerChecker = layer -> layer instanceof ActivationLayer;
        final LayerBlockConfig toTest = new Act();
        ProbingBuilderAdapter.testLayerBlock(layerChecker, toTest, 1);
    }
}