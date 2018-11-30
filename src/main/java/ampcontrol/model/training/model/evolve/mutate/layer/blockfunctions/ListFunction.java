package ampcontrol.model.training.model.evolve.mutate.layer.blockfunctions;

import ampcontrol.model.training.model.layerblocks.*;
import lombok.Builder;
import lombok.Singular;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.nd4j.linalg.activations.impl.ActivationReLU;

import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.function.IntUnaryOperator;

/**
 * Wraps as list of LayerBlockConfig factories. Selects one of them based on a provided indexSupplier
 *
 * @author Christian Skärby
 */
@Builder(builderClassName = "Builder")
public class ListFunction implements Function<Long, LayerBlockConfig> {

    private final IntUnaryOperator indexSupplier;
    @Singular("function") private final List<Function<Long, LayerBlockConfig>> functions;

    public ListFunction(IntUnaryOperator indexSupplier, List<Function<Long, LayerBlockConfig>> functions) {
        this.indexSupplier = indexSupplier;
        this.functions = functions;
    }

    @Override
    public LayerBlockConfig apply(Long nOut) {
        return functions.get(indexSupplier.applyAsInt(functions.size())).apply(nOut);
    }

    /**
     * Returns a ListFunctionBuilder which uses all 2D convolution blocks with equal probability
     * @param rng Random number generator
     * @return a ListFunctionBuilder
     */
    public static Builder allConv2D(Random rng) {

        return builder()
                .function(
                        nOut -> new Conv2D()
                                .setConvolutionMode(ConvolutionMode.Same)
                                .setNrofKernels(nOut.intValue())
                                .setKernelSize_w(1 + rng.nextInt(4) * 2)
                                .setKernelSize_h(1 + rng.nextInt(4) * 2)
                                .setActivation(new ActivationReLU()))
                .function(
                        nOut -> new Conv2DBatchNormAfter()
                                .setConvolutionMode(ConvolutionMode.Same)
                                .setNrofKernels(nOut.intValue())
                                .setKernelSize_w(1 + rng.nextInt(4) * 2)
                                .setKernelSize_h(1 + rng.nextInt(4) * 2)
                                .setActivation(new ActivationReLU()))
                .function(
                        nOut -> new Conv2DBatchNormBefore()
                                .setConvolutionMode(ConvolutionMode.Same)
                                .setNrofKernels(nOut.intValue())
                                .setKernelSize_w(1 + rng.nextInt(4) * 2)
                                .setKernelSize_h(1 + rng.nextInt(4) * 2)
                                .setActivation(new ActivationReLU()))
                .function(
                        nOut -> new Conv2DBatchNormBetween()
                                .setConvolutionMode(ConvolutionMode.Same)
                                .setNrofKernels(nOut.intValue())
                                .setKernelSize_w(1 + rng.nextInt(4) * 2)
                                .setKernelSize_h(1 + rng.nextInt(4) * 2)
                                .setActivation(new ActivationReLU()))
                .indexSupplier(rng::nextInt);
    }
}
