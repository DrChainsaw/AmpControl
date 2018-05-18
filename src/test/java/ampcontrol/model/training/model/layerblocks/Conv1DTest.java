package ampcontrol.model.training.model.layerblocks;

import org.deeplearning4j.nn.conf.layers.Convolution1DLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.junit.Test;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationCube;

import java.util.function.Function;

import static org.junit.Assert.assertTrue;

/**
 * Test cases for {@link Conv1D}
 *
 * @author Christian Skärby
 */
public class Conv1DTest {

    /**
     * Test that name contains all set parameters
     */
    @Test
    public void name() {
        final int nrofKernels = 666;
        final int kernelSize = 777;
        final int stride = 888;
        final IActivation act = new ActivationCube();
        String result = new Conv1D()
                .setNrofKernels(nrofKernels)
                .setKernelSize(kernelSize)
                .setStride(stride)
                .setActivation(act)
                .name();
        assertTrue("Parameter not part of name!", result.contains(String.valueOf(nrofKernels)));
        assertTrue("Parameter not part of name!", result.contains(String.valueOf(kernelSize)));
        assertTrue("Parameter not part of name!", result.contains(String.valueOf(stride)));
        assertTrue("Parameter not part of name!", result.contains(LayerBlockConfig.actToStr(act)));
    }

    /**
     * Test addLayers. Tests internals which is bad, but CBA to do anything else for such a simple class.
     */
    @Test
    public void addLayers() {
        final Function<Layer, Boolean> layerChecker = layer -> layer instanceof Convolution1DLayer;
        final LayerBlockConfig toTest = new Conv1D();
        ProbingBuilderAdapter.testLayerBlock(layerChecker, toTest, 1);
    }
}