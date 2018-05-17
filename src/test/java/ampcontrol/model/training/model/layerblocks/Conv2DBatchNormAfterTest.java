package ampcontrol.model.training.model.layerblocks;

import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.junit.Test;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationCube;

import java.util.function.Function;

import static org.junit.Assert.assertTrue;

/**
 * Test cases for {@link Conv2DBatchNormAfter}
 *
 * @author Christian Skärby
 */
public class Conv2DBatchNormAfterTest {

    /**
     * Test that name contains all set parameters
     */
    @Test
    public void name() {
        final int nrofKernels = 666;
        final int kernelSize = 777;
        final int stride_h = 888;
        final int stride_w = 999;
        final IActivation act = new ActivationCube();
        String result = new Conv2DBatchNormAfter()
                .setNrofKernels(nrofKernels)
                .setKernelSize(kernelSize)
                .setStride_h(stride_h)
                .setStride_w(stride_w)
                .setActivation(act)
                .name();
        assertTrue("Parameter not part of name!", result.contains(String.valueOf(nrofKernels)));
        assertTrue("Parameter not part of name!", result.contains(String.valueOf(kernelSize)));
        assertTrue("Parameter not part of name!", result.contains(String.valueOf(stride_h)));
        assertTrue("Parameter not part of name!", result.contains(String.valueOf(stride_w)));
        assertTrue("Parameter not part of name!", result.contains(LayerBlockConfig.actToStr(act)));
    }

    /**
     * Test that name contains all set parameters
     */
    @Test
    public void name2() {
        final int nrofKernels = 666;
        final int kernelSize_h = 777;
        final int kernelSize_w = 888;
        final int stride = 999;
        final IActivation act = new ActivationCube();
        String result = new Conv2DBatchNormAfter()
                .setNrofKernels(nrofKernels)
                .setKernelSize_h(kernelSize_h)
                .setKernelSize_w(kernelSize_w)
                .setStride(stride)
                .setActivation(act)
                .name();
        assertTrue("Parameter not part of name!", result.contains(String.valueOf(nrofKernels)));
        assertTrue("Parameter not part of name!", result.contains(String.valueOf(kernelSize_h)));
        assertTrue("Parameter not part of name!", result.contains(String.valueOf(kernelSize_w)));
        assertTrue("Parameter not part of name!", result.contains(String.valueOf(stride)));
        assertTrue("Parameter not part of name!", result.contains(LayerBlockConfig.actToStr(act)));
    }

    /**
     * Test addLayers. Tests internals which is bad, but CBA to do anything else for such a simple class.
     */
    @Test
    public void addLayers() {
        final Function<Layer, Boolean> layerChecker = new ProbingBuilderAdapter.FunctionQueue<>(
                Conv2DTest.conv2dLayerChecker,
                layer -> layer instanceof BatchNormalization);
        final LayerBlockConfig toTest = new Conv2DBatchNormAfter();
        ProbingBuilderAdapter.testLayerBlock(layerChecker, toTest, 2);
    }
}