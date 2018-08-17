package ampcontrol.model.training.model.layerblocks.adapters;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * Adapter for {@link ComputationGraphConfiguration.GraphBuilder}.
 *
 * @author Christian Skärby
 */
public class GraphAdapter implements GraphBuilderAdapter {

    private static final Logger log = LoggerFactory.getLogger(GraphBuilderAdapter.class);

    private final ComputationGraphConfiguration.GraphBuilder builder;

    public GraphAdapter(ComputationGraphConfiguration.GraphBuilder builder) {
        this.builder = builder;
    }

    @Override
    public LayerBlockConfig.BlockInfo layer(LayerBlockConfig.BlockInfo info, Layer layer) {
        final int layerInd = info.getPrevLayerInd()+1;
        final String thisName = info.getName(String.valueOf(layerInd));
        log.info("Create layer vertex " +thisName + " with input "+ Arrays.toString(info.getInputsNames()));
        builder.addLayer(thisName, layer, info.getInputsNames());
        return new LayerBlockConfig.SimpleBlockInfo.Builder(info)
                .setInputs(new String[] {thisName})
                .setPrevLayerInd(layerInd)
                .build();
    }

    @Override
    public GraphBuilderAdapter addLayer(String layerName, Layer layer, String... layerInputs) {
        builder.addLayer(layerName,layer,layerInputs);
        return this;
    }

    @Override
    public GraphBuilderAdapter addVertex(String vertexName, GraphVertex vertex, String... vertexInputs) {
        builder.addVertex(vertexName,vertex,vertexInputs);
        return this;
    }

    @Override
    public String mergeIfMultiple(String mergeName, String[] inputs) {
        if(inputs.length == 1) {
            return inputs[0];
        }
        builder.addVertex(mergeName, new MergeVertex(), inputs);
        return mergeName;
    }
}
