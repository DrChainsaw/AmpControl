package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.GlobPool;
import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import ampcontrol.model.training.model.vertex.ChannelMultVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Squeeze-exitation block. Add after any block which outputs CNN 2D features to create learnable channel gates (e.g.
 * turn down oyster-activation in presence of desert and sun activations).
 *
 * https://arxiv.org/abs/1709.01507
 *
 * @author Christian Skärby
 */
public class SeBlock implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(SeBlock.class);

    private double reduction = 16;
    private IActivation activation = new ActivationReLU();
    private LayerBlockConfig globPool = new GlobPool().setType(PoolingType.AVG);

    private boolean globPoolChanged = false;

    @Override
    public String name() {
        String actStr = LayerBlockConfig.actToStr(activation);
        actStr = actStr.isEmpty() ? actStr : "_" + actStr;
        String gpStr = globPoolChanged ? globPool.name() + "_" : "";
        String reduction = String.valueOf(this.reduction).replace(".", "p");
        return "se_" + gpStr + reduction + actStr;
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        throw new IllegalArgumentException("Can only stack graphs!");
    }

    @Override
    public BlockInfo addLayers(GraphBuilderAdapter graphBuilder, BlockInfo info) {
        BlockInfo nextLayer = info;
        log.info("Create squeeze-excite block " + info);

        int nrofOutputChannels = info.getPrevNrofOutputs();

        nextLayer = globPool.addLayers(graphBuilder, nextLayer);

        int nextLayerInd = nextLayer.getPrevLayerInd() + 1;
        final String squeeze = "squeeze" + nextLayerInd;
        graphBuilder.addLayer(squeeze,
                new DenseLayer.Builder()
                        .nOut((int)Math.max(nrofOutputChannels / reduction, 1))
                        .activation(activation)
                        .build(), nextLayer.getInputsNames());

        final String excite = "excite" + nextLayerInd;
        graphBuilder.addLayer(excite,
                new DenseLayer.Builder()
                        .nOut(nrofOutputChannels)
                        .activation(new ActivationSigmoid())
                        .build(), squeeze);

        final String inputsToScale = graphBuilder.mergeIfMultiple("seMv" + nextLayerInd, info.getInputsNames());

        String gate = "seGate" + nextLayerInd;
        log.info("chann receive vertex " + gate + " with inputs " + excite + " and " + info);
        graphBuilder.addVertex(gate,
                new ChannelMultVertex(),
                inputsToScale,
                excite);


        return new SimpleBlockInfo.Builder(nextLayer)
                .setInputs(new String[] {gate})
                .build();
    }

    /**
     * Number of nodes in dense layers will be nrofChannels / reduction
     * @param reduction the reduction
     * @return the {@link SeBlock}
     */
    public SeBlock setReduction(double reduction) {
        this.reduction = reduction;
        return this;
    }

    /**
     * Sets activation function to use.
     *
     * @param activation the activation function
     * @return the {@link SeBlock}
     */
    public SeBlock setActivation(IActivation activation) {
        this.activation = activation;
        return this;
    }

    /**
     * Sets the {@link LayerBlockConfig} to be used as global pool.
     * @param globPool the {@link LayerBlockConfig} to be used as global pool.
     * @return the {@link SeBlock}
     */
    public SeBlock setGlobPool(LayerBlockConfig globPool) {
        globPoolChanged = true;
        this.globPool = globPool;
        return this;
    }
}
