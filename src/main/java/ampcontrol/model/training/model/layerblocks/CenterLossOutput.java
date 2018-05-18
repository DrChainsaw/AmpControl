package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

/**
 * Adds a {@link CenterLossOutputLayer}.
 * 
 * @author Christian Skärby
 */
public class CenterLossOutput implements LayerBlockConfig {


    private final int nrofLabels;
    private double lambda = 0.0003;
    private double alpha = 0.5;

    /**
     * Contstructor
     * @param nrofLabels the number of labels to predict.
     */
    public CenterLossOutput(int nrofLabels) {
        this.nrofLabels = nrofLabels;
    }

    @Override
    public String name() {
        return ("clOut_" + lambda + "_" + alpha).replace('.', 'p');
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        final BlockInfo nextInfo = builder.layer(info, new CenterLossOutputLayer.Builder()
                .lossFunction(new LossMCXENT())
                .nOut(nrofLabels)
                .lambda(lambda)
                .alpha(alpha)
                .activation(new ActivationSoftmax())
                .build());

        return new SimpleBlockInfo.Builder(nextInfo)
                .setPrevNrofOutputs(nrofLabels)
                .build();
    }

    /**
     * Sets lambda, i.e how much of center loss to apply. Usually in range 0 - 0.1 (0 mean use softmax only).
     * @param lambda lambda factor to use
     * @return the {@link CenterLossOutput}
     */
    public CenterLossOutput setLambda(double lambda) {
        this.lambda = lambda; return this;
    }

    /**
     * Sets alpha, i.e. learning rate for center loss. Usually in range 0.01 - 1.
     * @param alpha alpha factor to use
     * @return the {@link CenterLossOutput}
     */
    public CenterLossOutput setAlpha(double alpha) {
        this.alpha = alpha; return this;
    }
}
