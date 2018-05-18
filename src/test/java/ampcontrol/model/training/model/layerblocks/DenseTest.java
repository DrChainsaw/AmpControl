package ampcontrol.model.training.model.layerblocks;


import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.junit.Test;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSELU;

import java.util.function.Function;

import static org.junit.Assert.assertTrue;

/**
 * Test cases for {@link Dense}
 *
 * @author Christian Skärby
 */
public class DenseTest {

    /**
     * Test name
     */
    @Test
    public void name() {
        final int width = 666;
        final IActivation act = new ActivationSELU();
        final String result = new Dense().setHiddenWidth(width).setActivation(act).name();
        assertTrue("Parameter not part of name!", result.contains(String.valueOf(width)));
        assertTrue("Parameter not part of name!", result.contains(LayerBlockConfig.actToStr(act)));
    }

    /**
     * Test addLayers. Tests internals which is bad, but CBA to do anything else for such a simple class.
     */
    @Test
    public void addLayers() {
        final Function<Layer, Boolean> layerChecker = layer -> layer instanceof DenseLayer;
        final LayerBlockConfig toTest = new Dense();
        ProbingBuilderAdapter.testLayerBlock(layerChecker, toTest, 1);
    }
}