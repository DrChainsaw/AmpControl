package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import org.deeplearning4j.nn.conf.distribution.BinomialDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.factory.Nd4j;

/**
 * "Dummy" output layer for testing.
 *
 * @author Christian Skärby
 */
public class DummyOutputLayer implements LayerBlockConfig {

    @Override
    public String name() {
        return "dummyOutput";
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        OutputLayer output = new OutputLayer.Builder()
                .nOut(info.getPrevNrofOutputs())
                .activation(new ActivationIdentity())
                .biasInit(0)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new BinomialDistribution(1,1)) // 100% probability of 1
                .build();
        return builder.layer(info,output);
    }

    /**
     * Sets weights output layer to the identity matrix.
     * @param graph the {@link ComputationGraph} to modify
     */
    public static void setEyeOutput(ComputationGraph graph) {
        final long[] wShape = graph.getOutputLayer(0).getParam("W").shape();
        graph.getOutputLayer(0).setParam("W", Nd4j.eye(wShape[0]));
    }
}
