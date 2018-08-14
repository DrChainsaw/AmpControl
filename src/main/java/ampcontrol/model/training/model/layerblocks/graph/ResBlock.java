package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.Conv2DBatchNormAfter;
import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Residual block. Changes the meaning of a given {@link LayerBlockConfig} so that it learns to produce a feature
 * residual, i.e. output = L(input) + input where L is the activation from the structure defined in blockConfig.
 *
 * https://arxiv.org/abs/1512.03385
 *
 * @author Christian Skärby
 */
public class ResBlock implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(ResBlock.class);

    private LayerBlockConfig blockConfig = new Conv2DBatchNormAfter();

    @Override
    public String name() {
        return "rb_" + blockConfig.name();
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        throw new IllegalArgumentException("Can only do graphs!");
    }

    @Override
    public BlockInfo addLayers(GraphBuilderAdapter graphBuilder, BlockInfo info) {

        log.info("Res block starting at " + info);
        BlockInfo nextLayer = blockConfig.addLayers(graphBuilder, info);

        final String add1 = graphBuilder.mergeIfMultiple("rbMv" + nextLayer.getPrevLayerInd(), info.getInputsNames());
        final String add2 = graphBuilder.mergeIfMultiple("rbMv" + (nextLayer.getPrevLayerInd()+1), nextLayer.getInputsNames());

        final String add = "rbAdd" + info.getPrevLayerInd();
        log.info("rb add: " + info + " and " +nextLayer);
        graphBuilder.addVertex(add,
                new ElementWiseVertex(ElementWiseVertex.Op.Add), add1, add2);

        return new SimpleBlockInfo.Builder(nextLayer)
                .setInputs(new String[] {add})
                .build();
    }

    /**
     * Sets the {@link LayerBlockConfig} which defines the structure for the residual features.
     * @param blockConfig The block which shall be treated as a residual
     * @return the {@link ResBlock}
     */
    public ResBlock setBlockConfig(LayerBlockConfig blockConfig) {
        this.blockConfig = blockConfig;
        return this;
    }

}
