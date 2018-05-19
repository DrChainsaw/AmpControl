package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adds an {@link OutputLayer}
 * 
 * @author Christian Skärby
 */
public class Output implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(Output.class);

    private final int nrofLabels;
    private INDArray weights = Nd4j.create(new double[]{1, 1, 1, 1});

    public Output(int nrofLabels) {
        this.nrofLabels = nrofLabels;
    }

    @Override
    public String name() {
        final String wString = weights.sum().getInt(0)== weights.length() ? "" : "_w_" + weights.toString()
                .replace("[", "")
                .replaceAll("\\.", "p")
                .replaceAll("0,\\s*", "_")
                .replace("0]", "");

        return "out" + wString;
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        log.info("Add output layer : " +info);
        BlockInfo nextInfo =  builder.layer(info,
                new OutputLayer.Builder()
                        .lossFunction(new LossMCXENT(weights))
                        .nOut(nrofLabels)
                        .activation(new ActivationSoftmax())
                        .build());
        return new SimpleBlockInfo.Builder(nextInfo)
                .setPrevNrofOutputs(nrofLabels)
                .build();
    }
}
