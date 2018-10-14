package ampcontrol.model.training.model.layerblocks.adapters;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.layers.Layer;

/**
 * Allows for spying on information about added layers through a {@link LayerSpy}.
 *
 * @author Christian Skärby
 */
public class LayerSpyAdapter implements GraphBuilderAdapter {

    private final GraphBuilderAdapter adapter;
    private final LayerSpy spy;

    /**
     * Interface from which layer info is obtained.
     */
    public interface LayerSpy {
        /**
         * Obtain information about added layers
         * @param layerName Name of added layer
         * @param layer {@link Layer} to add
         * @param layerInputs names of inputs to said layer
         */
        void accept(String layerName, Layer layer, String... layerInputs);
    }

    /**
     * Constructor
     * @param spy Will be notified of added layers
     * @param adapter {@link GraphBuilderAdapter} to spy on
     */
    public LayerSpyAdapter(LayerSpy spy, GraphBuilderAdapter adapter) {
        this.adapter = adapter;
        this.spy = spy;
    }

    @Override
    public GraphBuilderAdapter addLayer(String layerName, Layer layer, String... layerInputs) {
        spy.accept(layerName, layer, layerInputs);
        return adapter.addLayer(layerName, layer, layerInputs);
    }

    @Override
    public GraphBuilderAdapter addVertex(String vertexName, GraphVertex vertex, String... vertexInputs) {
        return adapter.addVertex(vertexName, vertex, vertexInputs);
    }

    @Override
    public String mergeIfMultiple(String mergeName, String[] inputs) {
        return adapter.mergeIfMultiple(mergeName, inputs);
    }

    @Override
    public LayerBlockConfig.BlockInfo layer(LayerBlockConfig.BlockInfo info, Layer layer) {
        final String[] inputs = info.getInputsNames();
        final LayerBlockConfig.BlockInfo outInfo = adapter.layer(info, layer);
        spy.accept(outInfo.getInputsNames()[0], layer, inputs);
        return outInfo;
    }
}
