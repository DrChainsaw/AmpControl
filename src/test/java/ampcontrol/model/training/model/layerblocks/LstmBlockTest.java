package ampcontrol.model.training.model.layerblocks;

import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.junit.Test;

import java.util.function.Function;

/**
 * Test cases for {@link LstmBlock}
 *
 * @author Christian Skärby
 */
public class LstmBlockTest {

    /**
     * Test addLayers. Tests internals which is bad, but CBA to do anything else for such a simple class.
     */
    @Test
    public void addLayers() {
        final Function<Layer, Boolean> layerChecker = layer -> layer instanceof LSTM;
        final LayerBlockConfig toTest = new LstmBlock();
        ProbingBuilderAdapter.testLayerBlock(layerChecker, toTest, 1);
    }
}