package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.GraphAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;

/**
 * Interface for a "Block" in a {@link ComputationGraphConfiguration}. Is expected to add layers or vertexes according
 * to provided {@link LayerBlockConfig.BlockInfo} and return {@link LayerBlockConfig.BlockInfo} describing the output.
 *
 * @author Christian Skärby
 */
public interface GraphBlockConfig {

    /**
     * Adds layers to a {@link org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder}
     * @param graphBuilder the builder
     * @param info contains info on what the inputs are
     * @return {@link LayerBlockConfig.BlockInfo} with outputs
     */
    default LayerBlockConfig.BlockInfo addLayers(ComputationGraphConfiguration.GraphBuilder graphBuilder, LayerBlockConfig.BlockInfo info) {
        return addLayers(new GraphAdapter(graphBuilder), info);
    }

    /**
     * Adds layers to a {@link GraphBuilderAdapter}
     * @param graphBuilder the builder
     * @param info contains info on what the inputs are
     * @return {@link LayerBlockConfig.BlockInfo} with outputs
     */
    LayerBlockConfig.BlockInfo addLayers(GraphBuilderAdapter graphBuilder, LayerBlockConfig.BlockInfo info);

}
