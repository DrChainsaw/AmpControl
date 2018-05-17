package ampcontrol.model.training.model.layerblocks.adapters;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.layers.Layer;

/**
 * Facade interface for {@link org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder} and
 * {@link org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder}.
 *
 * @author Christian Skärby
 */
public interface BuilderAdapter {

    /**
     * Adds a {@link Layer} according to the given {@link LayerBlockConfig.BlockInfo}. Returns
     * {@link LayerBlockConfig.BlockInfo} for the next layer.
     * @param info
     * @param layer
     * @return {@link LayerBlockConfig.BlockInfo} for the next layer
     */
    LayerBlockConfig.BlockInfo layer(LayerBlockConfig.BlockInfo info, Layer layer);

}
