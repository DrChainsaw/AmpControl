package ampcontrol.model.training.model.layerblocks.adapters;


import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.layers.Layer;

/**
 * Facade interface for {@link org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder}. Provides access
 * to graph specific functionality such as multiple inputs and non-layer vertexes.
 *
 * @author Christian Skärby
 */
public interface GraphBuilderAdapter extends BuilderAdapter {

    /**
     * Add a {@link Layer} with the specified name and specified inputs.
     *
     * @param layerName   Name/label of the layer to add
     * @param layer       The layer configuration
     * @param layerInputs Inputs to this layer (must be 1 or more). Inputs may be other layers, GraphVertex objects,
     *                    on a combination of the two.
     * @return The adapter instance
     */
    GraphBuilderAdapter addLayer(String layerName, Layer layer, String... layerInputs);

    /**
     * Add a {@link GraphVertex} with the specified name and specified inputs.
     *
     * @param vertexName
     * @param vertex
     * @param vertexInputs
     * @return The adapter instance
     */
    GraphBuilderAdapter addVertex(String vertexName, GraphVertex vertex, String... vertexInputs);

    /**
     * Merges the inputs if more than one.
     * @param mergeName the desired name of the output from the merge
     * @param inputs Array of inputs
     * @return the name of the input to next layer
     */
    String mergeIfMultiple(String mergeName, String[] inputs);

}
