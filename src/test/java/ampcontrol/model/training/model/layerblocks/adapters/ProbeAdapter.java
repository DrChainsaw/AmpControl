package ampcontrol.model.training.model.layerblocks.adapters;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.layers.Layer;

import java.util.ArrayList;
import java.util.List;

/**
 * {@link GraphBuilderAdapter} used for probing in tests
 *
 * @author Christian Skärby
 */
public class ProbeAdapter implements GraphBuilderAdapter {

    // yeah yeah yeah. Tests are simple enough...
    public final List<Object> inputs = new ArrayList<>();

    @Override
    public GraphBuilderAdapter addLayer(String layerName, Layer layer, String... layerInputs) {
        inputs.add(layerName);
        inputs.add(layer);
        inputs.add(layerInputs);
        return this;
    }

    @Override
    public GraphBuilderAdapter addVertex(String vertexName, GraphVertex vertex, String... vertexInputs) {
        inputs.add(vertexName);
        inputs.add(vertex);
        inputs.add(vertexInputs);
        return this;
    }

    @Override
    public String mergeIfMultiple(String mergeName, String[] inputs) {
        this.inputs.add(mergeName);
        this.inputs.add(inputs);
        return "";
    }

    @Override
    public LayerBlockConfig.BlockInfo layer(LayerBlockConfig.BlockInfo info, Layer layer) {
        inputs.add(info.getPrevLayerInd() + "");
        inputs.add(layer);
        inputs.add(info.getInputsNames());
        return info;
    }

}
