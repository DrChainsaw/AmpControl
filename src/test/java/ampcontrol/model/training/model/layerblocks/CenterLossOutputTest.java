package ampcontrol.model.training.model.layerblocks;

import org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.junit.Test;

import java.util.function.Function;

import static org.junit.Assert.assertNotEquals;

/**
 * Test cases for {@link CenterLossOutput}
 *
 * @author Christian Skärby
 */
public class CenterLossOutputTest {

    @Test
    public void name() {
        assertNotEquals("Names shall be unique", new CenterLossOutput(3).setAlpha(0.1), new CenterLossOutput(3).setAlpha(0.0001) );
        assertNotEquals("Names shall be unique", new CenterLossOutput(3).setLambda(0.1), new CenterLossOutput(3).setLambda(0.0001) );
    }

    /**
     * Test addLayers. Tests internals which is bad, but CBA to do anything else for such a simple class.
     */
    @Test
    public void addLayers() {
        final Function<Layer, Boolean> layerChecker = layer -> layer instanceof CenterLossOutputLayer;
        final LayerBlockConfig toTest = new CenterLossOutput(3);
        ProbingBuilderAdapter.testLayerBlock(layerChecker, toTest, 1);
    }
}