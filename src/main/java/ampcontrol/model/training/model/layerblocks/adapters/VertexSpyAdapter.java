package ampcontrol.model.training.model.layerblocks.adapters;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.layers.Layer;

/**
 * Allows for spying on information about added vertices through a {@link VertexSpy}.
 *
 * @author Christian Skärby
 */
public class VertexSpyAdapter implements GraphBuilderAdapter {

    private final GraphBuilderAdapter adapter;
    private final VertexSpy vertexSpy;

    /**
     * Interface from which vertex info is obtained.
     */
    public interface VertexSpy {
        /**
         * Obtain information about added layers
         * @param vertexName Name of added layer
         * @param vertex {@link Layer} to add
         * @param vertexInputs names of inputs to said layer
         */
        void accept(String vertexName, GraphVertex vertex, String... vertexInputs);
    }

    public VertexSpyAdapter(VertexSpy vertexSpy, GraphBuilderAdapter adapter) {
        this.adapter = adapter;
        this.vertexSpy = vertexSpy;
    }

    @Override
    public GraphBuilderAdapter addLayer(String layerName, Layer layer, String... layerInputs) {
        return adapter.addLayer(layerName, layer, layerInputs);
    }

    @Override
    public GraphBuilderAdapter addVertex(String vertexName, GraphVertex vertex, String... vertexInputs) {
        vertexSpy.accept(vertexName, vertex, vertexInputs);
        return adapter.addVertex(vertexName, vertex, vertexInputs);
    }

    @Override
    public String mergeIfMultiple(String mergeName, String[] inputs) {
        return adapter.mergeIfMultiple(mergeName, inputs);
    }

    @Override
    public LayerBlockConfig.BlockInfo layer(LayerBlockConfig.BlockInfo info, Layer layer) {
        return adapter.layer(info, layer);
    }
}
