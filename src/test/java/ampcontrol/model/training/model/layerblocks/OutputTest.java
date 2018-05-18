package ampcontrol.model.training.model.layerblocks;

import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.junit.Test;

import java.util.function.Function;

/**
 * Test cases for {@link Output}
 *
 * @author Christian Skärby
 */
public class OutputTest {

    /**
     * Test addLayers. Tests internals which is bad, but CBA to do anything else for such a simple class.
     */
    @Test
    public void addLayers() {
        final Function<Layer, Boolean> layerChecker = layer -> layer instanceof OutputLayer;
        final LayerBlockConfig toTest = new Output(3);
        ProbingBuilderAdapter.testLayerBlock(layerChecker, toTest, 1);
    }

}